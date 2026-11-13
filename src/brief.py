"""
Phase 5 - The pre-meeting brief for Team TS3012 (or any team).

    python src/brief.py              TS3012, written into answers.json and brief.md
    python src/brief.py ES2015       any other team (brief.md only)

"Before their final meeting" means the agent may only see meetings BEFORE the last one
(TS3012: a, b, c). That rule is enforced in CODE, not just asked for in the prompt:
  * tools.run_tool(up_to_meeting="c"): get_memory hides meeting d, search only searches a-c,
    read_turns refuses meeting d
  * every citation from a later meeting is rejected by check_and_repair
So the brief can't "see the future", even if the model tried.

Same loop as the question agent (agent.py), with a different final tool: submit_brief, whose
points are grouped into the four sections Catalist asked for. Code checks every point's
citations; points that fail are sent back once, and any point still without a valid citation is
dropped, so every line in the brief is backed by an exact quote.
"""

import json
import sys
import time

from openai import OpenAI

import re

from agent import MODEL, PRICES, check_and_repair, check_roles, model_settings
from memory import get_memory
from parse import ROOT
from prompts import BRIEF_PROMPT_VERSION, BRIEF_SYSTEM, brief_user_message
from search import list_meetings
from tools import TOOL_SPECS, run_tool

MAX_STEPS = 20
SECTIONS = [("decided", "What has been decided"),
            ("changed", "What changed along the way"),
            ("open_actions", "Open actions and owners"),
            ("open_disagreements", "Disagreements still open")]

_POINT = {"type": "object", "properties": {
    "point": {"type": "string", "description": "One plain sentence."},
    "citations": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
        "type": "object", "properties": {
            "meeting_id": {"type": "string"},
            "quote": {"type": "string", "description": "6+ consecutive words copied exactly from ONE turn"}},
        "required": ["meeting_id", "quote"]}}},
    "required": ["point", "citations"]}
SUBMIT_BRIEF = {"type": "function", "function": {
    "name": "submit_brief", "description": "Submit the brief. Every point needs 1-2 exact citations.",
    "parameters": {"type": "object", "properties": {k: {"type": "array", "items": _POINT} for k, _ in SECTIONS},
                   "required": [k for k, _ in SECTIONS]}}}
BRIEF_TOOLS = [t for t in TOOL_SPECS if t["function"]["name"] != "submit_answer"] + [SUBMIT_BRIEF]


GENERIC = {"remote", "control", "design", "product", "type", "user", "interface", "team", "general"}


def _keywords(topic: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", topic.lower())) - GENERIC
    words = {w for w in words if len(w) > 3}
    return words | ({"color"} if "colour" in words else set())


def memory_checklist(team: str, cutoff: str) -> tuple[list[str], str]:
    """CODE builds a checklist from the extracted memory, so the brief can't silently skip a
    settled decision. Added after brief run 1: memory said energy = regular batteries (c #75),
    scroll-wheel rejected (c #635) and colour = yellow (c #593), and the brief mentioned none of them.
    Returns (topics that must be covered, text shown to the model)."""
    facts = get_memory(team, up_to_ord="abcd".index(cutoff) + 1)
    settled = []
    for f in facts:     # current, settled decisions from meeting b onwards (meeting a is mostly the brief)
        if f["current"] and f["type"] == "decision" and f["status"] in ("agreed", "rejected") \
                and f["ord"] >= 2 and f["topic"] not in settled and _keywords(f["topic"]):
            settled.append(f["topic"])
    actions = [f for f in facts if f["type"] == "action_item" and f["current"] and f["status"] == "open"]
    text = ("Checklist from the extracted memory (made by code):\n"
            "Settled decisions to cover (in 'decided', or in 'changed' if they changed): "
            + "; ".join(settled)
            + "\nAction items still marked open in memory (check each one: was it done, reported back or "
              "ruled out in a later meeting? Only list it as open if not):\n"
            + "\n".join(f"- {f['meeting_id']}: {f['summary']}" for f in actions))
    return settled, text


def missing_topics(args: dict, settled: list[str]) -> list[str]:
    text = " ".join(p.get("point", "") for key, _ in SECTIONS for p in (args.get(key) or [])).lower()
    return [t for t in settled if not any(k in text for k in _keywords(t))]


def check_brief(args: dict, team: str, cutoff: str) -> tuple[dict, list[str]]:
    """CODE checks every point: citations (exact, right team, not after the cutoff) and roles."""
    kept, problems = {}, []
    for key, _ in SECTIONS:
        kept[key] = []
        for i, p in enumerate(args.get(key, []) or []):
            good, failures, _ = check_and_repair(p.get("citations", []), team, cutoff)
            roles = check_roles(p.get("point", ""), team)
            if good and not roles:
                kept[key].append({"point": p["point"].strip(), "citations": good})
            label = f"{key}[{i}] '{p.get('point', '')[:60]}'"
            problems += [f"{label}: {f}" for f in failures] + [f"{label}: {r}" for r in roles]
            if not good and not failures:
                problems.append(f"{label}: no citation")
    return kept, problems


def render(team: str, cutoff: str, brief: dict) -> str:
    """Plain text with the citations inline, so every claim can be checked where it is made."""
    lines = [f"Brief for Team {team}, before their final meeting "
             f"(based on meetings up to {team}{cutoff})."]
    for key, title in SECTIONS:
        lines.append(f"\n{title}:")
        items = brief.get(key, [])
        if not items:
            lines.append("- Nothing found with evidence.")
        for p in items:
            cites = "; ".join(f'{c["meeting_id"]}: "{c["quote"]}"' for c in p["citations"])
            lines.append(f"- {p['point']} [{cites}]")
    return "\n".join(lines)


def write_brief(team: str = "TS3012") -> dict:
    meetings = list_meetings(team)
    if len(meetings) < 2:
        raise SystemExit(f"{team} needs at least 2 meetings for a brief.")
    cutoff = meetings[-2]["letter"]          # last meeting BEFORE the final one
    client = OpenAI()
    settled, checklist = memory_checklist(team, cutoff)
    messages = [{"role": "system", "content": BRIEF_SYSTEM},
                {"role": "user", "content": brief_user_message(team, cutoff) + "\n\n" + checklist}]
    trace, tin, tout, fix_rounds, final, t0 = [], 0, 0, 0, None, time.time()
    print(f"Brief for {team}: meetings up to {team}{cutoff} only, model {MODEL}")

    for step in range(1, MAX_STEPS + 2):
        last = step > MAX_STEPS
        extra = model_settings(MODEL)
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=BRIEF_TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit_brief"}} if last else "auto", **extra)
        tin += resp.usage.prompt_tokens
        tout += resp.usage.completion_tokens
        msg = resp.choices[0].message
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                       for c in msg.tool_calls]
        messages.append(assistant)
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Continue with tools; finish with submit_brief."})
            continue

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if call.function.name != "submit_brief":
                result = run_tool(call.function.name, args, team, up_to_meeting=cutoff)
                trace.append({"step": step, "tool": call.function.name, "args": args, "result_chars": len(result)})
                print(f"  step {step}: {call.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items() if k != 'team')})"
                      f" -> {len(result):,} chars")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                continue
            kept, problems = check_brief(args, team, cutoff)
            gaps = missing_topics(args, settled)
            if gaps and fix_rounds == 0:
                problems.append("These settled decisions are missing from the brief: " + "; ".join(gaps)
                                + ". Add each (with a citation) unless it is truly unimportant.")
            n_points = sum(len(v) for v in kept.values())
            print(f"  step {step}: submit_brief -> {n_points} points kept, {len(problems)} problem(s)")
            trace.append({"step": step, "tool": "submit_brief", "kept": n_points, "problems": problems})
            if problems and fix_rounds < 2 and not last:
                fix_rounds += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "content":
                                 "Not accepted yet. Fix these (use read_turns to copy exact words), then call "
                                 "submit_brief again with ALL points:\n- " + "\n- ".join(problems[:20])})
                continue
            final = kept                      # points still failing are dropped
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "Accepted."})
        if final is not None:
            break

    final = final or {k: [] for k, _ in SECTIONS}
    text = render(team, cutoff, final)
    p_in, p_out = PRICES.get(MODEL, (0, 0))
    cost = tin / 1e6 * p_in + tout / 1e6 * p_out
    log = {"team": team, "cutoff": cutoff, "model": MODEL, "prompt_version": BRIEF_PROMPT_VERSION,
           "brief": final, "text": text, "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 5),
           "seconds": round(time.time() - t0, 1), "trace": trace}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / f"brief_{team}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / f"brief_{team}.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n{text}\n\n{sum(1 for t in trace if t['tool'] != 'submit_brief')} tool calls, "
          f"{tin:,}+{tout:,} tokens, ${cost:.4f}, {log['seconds']}s")
    return log


if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "TS3012"
    result = write_brief(team)
    if team == "TS3012":                      # the brief Catalist asked for goes into answers.json
        path = ROOT / "answers.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"answers": []}
        data["brief"] = result["text"]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Written to answers.json (\"brief\") and brief_TS3012.md")