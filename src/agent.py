"""
Phase 4 - The agent: a hand-written tool-calling loop (no framework).

    python src/agent.py TS3012 "What case material did they choose and why?"

How one question is answered:
  1. The LLM gets the question, the team, and the tool list.
  2. LOOP (max MAX_STEPS rounds): the LLM picks one or more tool calls ->
     code runs them (tools.py) -> the small results go back to the LLM.
  3. The LLM finishes by calling submit_answer(answer, citations).
  4. CODE checks every citation (verify.py). A quote that is close but not exact
     is repaired to the real words; one that can't be repaired is sent back to the
     LLM with the reason, and it gets up to MAX_FIX_ROUNDS tries to fix it.
     Citations that still fail are removed: the grader never sees an unchecked quote.
  5. Every step is logged (logs/<id>.json): which tool, which arguments, how big
     the result was. That trace is the "real agent behaviour" evidence.

Model vs code: the LLM decides what to look for, what the evidence means, and
how to phrase the answer. Code runs every lookup, enforces the team / meeting
guardrails and the step limit, and has the final say on every quote.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from extract import repair_evidence
from parse import ROOT, TRANSCRIPTS_DIR, parse_turns
from prompts import AGENT_PROMPT_VERSION, AGENT_SYSTEM, agent_user_message
import re

from memory import get_memory
from tools import TOOL_SPECS, get_roles, run_tool
from verify import check_citation
# Checks added after the first run on Catalist's 25 questions (Q16 came back with no citations and
# turn numbers in the text, Q25 had 4 sentences, 9 answers cited a single meeting).

load_dotenv(ROOT / ".env")
MODEL = os.getenv("AGENT_MODEL", "gpt-6-luna")   # was gpt-4.1-mini; set AGENT_MODEL to compare
MAX_STEPS = 12            # LLM rounds before we force an answer
MAX_FIX_ROUNDS = 2        # chances to fix failing citations
LOG_DIR = ROOT / "logs"
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-5-mini": (0.25, 2.00),
          "gpt-6-luna": (0.10, 0.50), "gpt-5.6-luna": (0.20, 1.20)}   # $ per 1M tokens (OpenAI pricing page, Nov 2026)


def model_settings(model: str) -> dict:
    """Per-model API settings. gpt-6 / luna models are reasoning models; on Chat Completions they
    only support function calling with reasoning_effort 'none' (OpenAI model docs), and they don't
    take temperature. gpt-5 / o-series don't take temperature either. Everything else: temperature 0."""
    if model.startswith("gpt-6") or "luna" in model:
        return {"extra_body": {"reasoning_effort": "none"}}
    if model.startswith(("gpt-5", "o")):
        return {}
    return {"temperature": 0}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "team")


def check_and_repair(citations: list[dict], team: str, up_to_meeting: str | None) -> tuple[list, list, list]:
    """CODE decides which citations are real. Returns (good, failures for the LLM, notes for the log)."""
    good, failures, notes = [], [], []
    for c in citations[:3]:
        mid, quote = str(c.get("meeting_id", "")).strip(), str(c.get("quote", "")).strip()
        if not mid.startswith(team):
            failures.append(f"{mid}: not a meeting of team {team}")
            continue
        if up_to_meeting and "abcd".index(mid[-1]) > "abcd".index(up_to_meeting):
            failures.append(f"{mid}: after the allowed meeting {team}{up_to_meeting}")
            continue
        ok, msg = check_citation(mid, quote)
        if ok:
            good.append({"meeting_id": mid, "quote": quote})
            continue
        # Close but not exact (a dropped "uh", different punctuation)? Use the real words instead.
        path = TRANSCRIPTS_DIR / f"{mid}.txt"
        repaired = repair_evidence(quote, parse_turns(path), -10) if path.exists() else None   # -10: no turn hint, search the whole meeting
        if repaired and len(repaired[1].split()) >= 6 and repaired[2] >= 0.85:
            good.append({"meeting_id": mid, "quote": repaired[1]})
            notes.append(f"repaired quote in {mid} ({repaired[2]:.2f}): {quote[:50]!r} -> {repaired[1][:50]!r}")
        else:
            failures.append(f"{mid}: {quote[:80]!r} -> {msg}")
    return good, failures, notes


def check_roles(answer: str, team: str) -> list[str]:
    """CODE checks every 'Speaker X (Role)' in the answer against get_roles (stated or deduced).
    Added after eval run 2: the agent called ES2005's C 'Industrial Designer' although
    get_roles said C = Project Manager. A wrong role is a factual error, so it is not accepted."""
    known = {}
    for line in get_roles(team).splitlines():
        m = re.match(r"Speaker ([A-D]) = ([A-Za-z ]+)", line)
        if m:
            known[m.group(1)] = m.group(2).strip().lower()
    problems = []
    for letter, role in re.findall(r"Speaker ([A-D]) \(([^)]+)\)", answer):
        if letter in known and role.split()[0].lower() not in known[letter]:
            problems.append(f"Speaker {letter} is the {known[letter].title()} according to get_roles, "
                            f"not '{role}'")
        if letter not in known:
            problems.append(f"Speaker {letter}'s role is not known: write just 'Speaker {letter}'")
    return problems


# ---------------------------------------------------------------- fix 1: one search per part
# Eval logs showed gpt-4.1-mini stopping after ~6-9 tool calls on 4-5 part questions, so one
# part often had no evidence. Now the FIRST step is forced: the model splits the question into
# parts with one search query each, and CODE runs every one of those searches. The model still
# decides the queries, reads the results and writes the answer; it just can't skip a part.
PLAN_TOOL = {"type": "function", "function": {
    "name": "plan_searches",
    "description": ("First step: split the question into its parts and give one short search query per part. "
                    "Code runs all the searches and returns the passages."),
    "parameters": {"type": "object", "properties": {
        "parts": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object", "properties": {
                "part": {"type": "string", "description": "e.g. 'what they chose in the end'"},
                "query": {"type": "string", "description": "2-6 focused words, e.g. 'final energy source batteries'"}},
            "required": ["part", "query"]}}},
        "required": ["parts"]}}}


def run_plan(parts: list[dict], team: str, up_to_meeting: str | None) -> str:
    """CODE runs one search per part (3 passages each, shorter, to keep the context small).
    Goes through run_tool, so the team lock and the brief cutoff still apply."""
    out = []
    for p in parts[:5]:
        res = run_tool("search", {"team": team, "query": str(p.get("query", "")), "k": 3, "chars": 1200},
                       team, up_to_meeting)
        out.append(f"=== Part: {p.get('part', '')}  (search: {p.get('query', '')!r})\n{res}")
    return "\n\n".join(out) + ("\n\nThese are starting points. Read around the key turns (read_turns), "
                               "check get_memory for what changed, and search again where a part is still unclear.")


# ---------------------------------------------------------------- fix 2: outcome claims vs memory
# T08 (final run) said IS1004's final remote had "an LCD screen"; memory already had
# "no LCD" as the current, settled decision. Nothing compared the two. Now CODE compares
# every outcome sentence ("final", "decided", "kept"...) with the team's current settled
# decisions: if the answer says something is IN but memory says REJECTED (or the reverse),
# the agent is shown the memory fact once and asked to check. The model makes the final call,
# because memory can be wrong too (extraction drops ~6% of facts and misses some proposals).
OUTCOME = re.compile(r"\b(final|finally|in the end|ultimately|eventually|decided|chose|chosen|opted|kept|"
                     r"went with|settled|agreed|ended up|end up|dropped|rejected|scrapped|instead)\b", re.I)
NEG_BEFORE = {"no", "not", "never", "without", "against", "drop", "dropped", "dropping", "reject", "rejected",
              "rejecting", "scrap", "scrapped", "remove", "removed", "abandon", "abandoned", "instead", "lose",
              "cut", "exclude", "excluded", "forget", "nor", "than"}
NEG_AFTER = {"dropped", "rejected", "scrapped", "removed", "abandoned", "cut", "excluded", "ruled", "discarded"}
TOPIC_STOP = {"remote", "control", "controls", "design", "product", "type", "user", "interface", "team",
              "general", "screen", "the", "and", "for", "use", "with", "final", "project", "case", "material",
              "feature", "features", "option", "options", "new", "button", "buttons", "placement", "choice",
              "on", "of", "in", "to", "a", "energy", "source", "power", "cost", "costs", "price", "budget",
              "colour", "color", "shape", "size"}   # too general: they match many unrelated sentences
HISTORY = re.compile(r"\b(first|initially|originally|provisionally|at first|earlier|considered|proposed|"
                     r"suggested|wanted|wished|thought)\b", re.I)   # sentence about an earlier stage
RESET = {"kept", "keep", "chose", "chosen", "used", "opted", "went", "but", "while", "whereas",
         "favour", "favor", "favoured", "favored"}   # a new positive clause starts: stop looking back


def _topic_words(topic: str) -> list[str]:
    topic = re.sub(r"(?<![a-z])((?:[a-z]_)+)", lambda m: m.group(1).replace("_", ""), topic.lower())  # l_c_d_ -> lcd
    return [w for w in re.findall(r"[a-z]+", topic) if w not in TOPIC_STOP and len(w) >= 3]


def _same(word: str, kw: str) -> bool:
    return word == kw or (len(kw) >= 5 and word[:5] == kw[:5])      # wheel ~ wheels, recognise ~ recognition


def check_outcomes(answer: str, team: str, up_to_meeting: str | None) -> list[str]:
    facts = [f for f in get_memory(team, up_to_ord="abcd".index(up_to_meeting) + 1 if up_to_meeting else None)
             if f["type"] == "decision" and f["current"] and f["status"] in ("agreed", "rejected")]
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if not OUTCOME.search(sentence) or HISTORY.search(sentence):
            continue                                    # not a claim about the outcome
        tokens = re.findall(r"[a-z]+", re.sub(r"L_C_D_|L_C_D", "lcd", sentence).lower())
        for f in facts:
            if " and " in f["topic"]:
                continue                                # mixed topic ("screen and speech recognition"): no single status
            kws = _topic_words(f["topic"])
            hits = [i for i, t in enumerate(tokens) if any(_same(t, k) for k in kws)]
            matched = {k for k in kws if any(_same(tokens[i], k) for i in hits)}
            if not kws or len(matched) < min(2, len(kws)):
                continue
            i = hits[0]
            before = []
            for t in reversed(tokens[max(0, i - 5):i]):  # look back up to 5 words, within the same clause
                if t in RESET:
                    break
                before.append(t)
            negated = (any(t in NEG_BEFORE for t in before)
                       or any(t in NEG_AFTER for t in tokens[i + 1:i + 5]))
            says = "left out / rejected" if negated else "included / chosen"
            if (f["status"] == "rejected") != negated:
                problems.append(f"Your sentence \"{sentence.strip()[:120]}\" reads as: {f['topic']} {says}. "
                                f"Memory's CURRENT decision is {f['status'].upper()} ({f['meeting_id']} #{f['turn']}): "
                                f"{f['summary']}")
    return problems[:3]


_ID = r"[A-Z]{2}\d{4}[a-d]"


def clean_answer(text: str) -> str:
    """CODE removes citation markers the model writes into the answer text. gpt-6-luna sometimes adds
    its own markers - hidden private-use characters ("\ue200cite\ue202TS3006b\ue201"), "[[ES2004c]]",
    "[ES2008a]", "[1][2][3]" or a trailing "Citations: ..." - although citations go in their own field.
    Catalist asks for plain sentences, so these are stripped by fixed rules (no model involved)."""
    text = re.sub("\ue200.*?\ue201", "", text)                     # hidden citation spans
    text = re.sub("[\ue000-\uf8ff]", "", text)                      # any other private-use characters
    text = re.sub(r"\s*Citations?:\s*(\[+" + _ID + r"\]+[\s,;]*)*$", "", text)
    text = re.sub(r"\s*\[+" + _ID + r"\]+", "", text)                # [ES2008a], [[ES2004c]]
    text = re.sub(r"\s*\[\d+\]", "", text)                           # [1][2][3]
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def salvage_quotes(answer: str, team: str, up_to_meeting: str | None) -> tuple[list, list]:
    """If the model never filled in citations but quoted the transcript inside its answer, CODE finds
    which meeting each quote is from and checks it with the grader's rules. Only exact (or repaired,
    >= 0.85 similar, 6+ words) quotes are kept, so this can't invent a citation."""
    from search import list_meetings
    cutoff = "abcd".index(up_to_meeting) + 1 if up_to_meeting else 9
    meetings = [m["meeting_id"] for m in list_meetings(team) if m["ord"] <= cutoff]
    good, notes = [], []
    for q in re.findall(r'"([^"]{25,})"', answer):
        for mid in meetings:
            g, _, n = check_and_repair([{"meeting_id": mid, "quote": q}], team, up_to_meeting)
            if g:
                good += g
                notes.append(f"salvaged a quote from the answer text: {mid}")
                break
        if len(good) == 3:
            break
    return good, notes


def answer_question(question: str, team: str, qid: str = "adhoc", model: str | None = None,
                    up_to_meeting: str | None = None, system: str = AGENT_SYSTEM,
                    user_message: str | None = None, verbose: bool = True,
                    plan_first: bool = True) -> dict:
    model = model or MODEL
    tools = [PLAN_TOOL] + TOOL_SPECS if plan_first else TOOL_SPECS
    outcome_checked = False
    client = OpenAI()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_message or agent_user_message(question, team)}]
    trace, tokens_in, tokens_out, fix_rounds = [], 0, 0, 0
    final, t0 = None, time.time()
    log = (lambda s: print(f"  [{qid}] {s}")) if verbose else (lambda s: None)

    for step in range(1, MAX_STEPS + 2):
        last = step > MAX_STEPS
        extra = model_settings(model)
        if last:                       # final round: code forces an answer, so the loop always ends
            choice = {"type": "function", "function": {"name": "submit_answer"}}
        elif step == 1 and plan_first:  # first round: code forces the plan (fix 1)
            choice = {"type": "function", "function": {"name": "plan_searches"}}
        else:
            choice = "auto"
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=choice, **extra)
        tokens_in += resp.usage.prompt_tokens
        tokens_out += resp.usage.completion_tokens
        msg = resp.choices[0].message
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                       for c in msg.tool_calls]
        messages.append(assistant)

        if not msg.tool_calls:                        # it wrote text instead of using a tool
            messages.append({"role": "user", "content": "Continue with tools; finish with submit_answer."})
            trace.append({"step": step, "note": "no tool call", "text": (msg.content or "")[:200]})
            continue

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "plan_searches":
                result = run_plan(args.get("parts", []), team, up_to_meeting)
                trace.append({"step": step, "tool": name, "args": args, "result_chars": len(result),
                              "result_preview": result[:300]})
                log(f"step {step}: plan_searches({len(args.get('parts', []))} parts: "
                    f"{[p.get('query') for p in args.get('parts', [])]}) -> {len(result):,} chars")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                continue
            if name != "submit_answer":
                result = run_tool(name, args, team, up_to_meeting)
                trace.append({"step": step, "tool": name, "args": args, "result_chars": len(result),
                              "result_preview": result[:300]})
                log(f"step {step}: {name}({_short(args)}) -> {len(result):,} chars")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                continue

            # ---- submit_answer: code checks the citations and the roles
            good, failures, notes = check_and_repair(args.get("citations", []), team, up_to_meeting)
            role_problems = check_roles(args.get("answer", ""), team)
            not_found = [p.get("part", "") for p in args.get("parts", [])
                         if "not found" in str(p.get("evidence", "")).lower()]
            trace.append({"step": step, "tool": "submit_answer", "answer": args.get("answer", ""),
                          "parts": args.get("parts", []), "valid": len(good), "failures": failures,
                          "repairs": notes, "role_problems": role_problems, "parts_not_found": not_found})
            log(f"step {step}: submit_answer -> {len(good)} valid citation(s), {len(failures)} failed"
                + (f", {len(notes)} repaired" if notes else "")
                + (f", {len(role_problems)} role error(s)" if role_problems else "")
                + (f", parts not found: {not_found}" if not_found else ""))
            problems = []
            answer_text = args.get("answer", "")
            if len(re.findall(r"[.!?](?:\s|$)", answer_text)) > 3:
                problems.append("The answer must be 1-3 sentences; shorten it.")
            if re.search(r"#\d", answer_text):
                problems.append("Don't put turn numbers (#123) in the answer text; evidence goes in citations.")
            if not args.get("citations"):
                # T04 (final eval): the model sent an EMPTY citations list 4 times and wrote the quotes
                # inside the answer instead; the old message didn't say why, so it never fixed it.
                problems.append("Your 'citations' list is EMPTY. Quotes go in the citations field as "
                                "{meeting_id, quote}, NOT inside the answer text. Remove the quotes from the "
                                "answer (1-3 plain sentences) and give 1-3 citations.")
            elif not good:
                problems.append("No citation passed the check. Every answer needs at least one valid citation.")
            elif len({c["meeting_id"] for c in good}) < 2 and fix_rounds == 0:
                problems.append("All citations come from one meeting, but these questions span 2+ meetings: "
                                "if another meeting supports part of the answer, cite it too.")
            if re.search(r'"[^"]{25,}"', answer_text):
                problems.append("Don't quote transcript text inside the answer; put it in citations.")
            if failures:
                problems.append("These citations failed the exact-quote check (use read_turns and copy "
                                "the exact words from one turn):\n- " + "\n- ".join(failures))
            if role_problems:
                problems.append("Wrong speaker roles:\n- " + "\n- ".join(role_problems))
            if not outcome_checked:                # fix 2: outcome claims vs memory, once per question
                outcome_checked = True
                contradictions = check_outcomes(answer_text, team, up_to_meeting)
                trace[-1]["outcome_check"] = contradictions
                if contradictions:
                    log(f"step {step}: outcome check -> {len(contradictions)} possible contradiction(s) with memory")
                    problems.append(
                        "Code compared your outcome claims with the team's current decisions in memory:\n- "
                        + "\n- ".join(contradictions)
                        + "\nCheck the later evidence (get_memory with that topic, read_turns). If memory is right, "
                          "fix the answer. If your sentence is about an earlier stage, make that clear. If you read "
                          "later evidence that memory missed, keep your answer.")
            if not_found and fix_rounds == 0:      # one extra chance to look for missing parts
                problems.append("You marked these parts as not found: " + "; ".join(not_found)
                                + ". Search once more for each (other words, later meetings) before giving up.")
            # one extra round when there is still no valid citation at all
            limit = MAX_FIX_ROUNDS + (1 if not good else 0)
            if problems and fix_rounds < limit and not last:
                fix_rounds += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "content":
                                 "Not accepted yet.\n" + "\n".join(problems)
                                 + "\nFix this, then call submit_answer again with the full answer and all citations."})
                continue
            if not good:                           # last resort: rescue quotes written in the answer text
                good, notes2 = salvage_quotes(answer_text, team, up_to_meeting)
                trace[-1]["salvaged"] = notes2
            final = {"answer": clean_answer(args.get("answer", "")), "citations": good,
                     "dropped_citations": failures}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "Accepted."})
        if final:
            break

    if final is None:
        final = {"answer": "", "citations": [], "dropped_citations": ["no answer submitted"]}
    p_in, p_out = PRICES.get(model, (0, 0))
    result = {"question_id": qid, "team": team, "question": question, **final,
              "model": model, "prompt_version": AGENT_PROMPT_VERSION,
              "steps": sum(1 for t in trace if "tool" in t), "llm_rounds": step,
              "tokens_in": tokens_in, "tokens_out": tokens_out,
              "cost_usd": round(tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out, 5),
              "seconds": round(time.time() - t0, 1), "trace": trace}
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{qid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit()
    r = answer_question(sys.argv[2], sys.argv[1])
    print("\nANSWER:", r["answer"])
    for c in r["citations"]:
        print(f"  [{c['meeting_id']}] \"{c['quote']}\"")
    if r["dropped_citations"]:
        print("  dropped:", r["dropped_citations"])
    print(f"\n{r['steps']} tool calls, {r['tokens_in']:,}+{r['tokens_out']:,} tokens, "
          f"${r['cost_usd']:.4f}, {r['seconds']}s  (log: logs/adhoc.json)")