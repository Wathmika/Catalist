"""
Phase 4 - Score the agent on our own 10 test questions (tests/my_questions.json).

    python tests/eval_agent.py                 run the agent, then score
    python tests/eval_agent.py T04 T07         only some questions
    python tests/eval_agent.py --rejudge       re-score the saved answers (no agent run)

Three scores per question:
  facts      how many of the human-written key facts the answer contains.
             Judged by an LLM (gpt-4.1-mini), because "C proposed kinetic" and
             "the Industrial Designer suggested kinetic energy" mean the same thing
             and no string match can see that. The judge only COMPARES the answer
             with our human-written key facts; it never decides what is true.
             Every judgement is printed, so a human can overrule it.
  citations  valid citations (code) and how many meetings they span (questions need 2+).
  evidence   do the cited turns land near the turns our own answer key cites? (code)

Results are saved to tests/results_<model>.json, so models / prompt versions can be compared.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from openai import OpenAI  # noqa: E402

from agent import MODEL, answer_question  # noqa: E402
from search import _connect  # noqa: E402

JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_SYSTEM = """You compare an answer with a list of key facts from a human-written answer key.
For each key fact, decide whether the answer EXPLICITLY states it. Wording may differ, and a
speaker may be named by letter or by role. Be strict:
- every specific detail in the fact (number, speaker, role, meeting, reason) must be in the answer;
- if the answer gives a speaker a different role than the fact, the fact is false;
- if the answer says something that contradicts the fact, the fact is false;
- a fact only implied, vague, or partly covered is false.
Return JSON only."""
# Run 1 showed why "strict" had to be spelled out: the first judge counted "young users like the
# spongy feel" as present in an answer that never mentioned it, and accepted a wrong role.


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def turn_of(meeting_id: str, quote: str) -> int | None:
    with _connect() as db:
        for r in db.execute("SELECT turn_idx, text FROM turns WHERE meeting_id = ?", (meeting_id,)):
            if norm(quote) in norm(r["text"]):
                return r["turn_idx"]
    return None


def judge(answer: str, key_facts: list[str]) -> list[bool]:
    schema = {"type": "object", "additionalProperties": False, "required": ["present"],
              "properties": {"present": {"type": "array", "items": {"type": "boolean"}}}}
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(key_facts))
    resp = OpenAI().chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": f"Key facts:\n{facts}\n\nAnswer:\n{answer}\n\n"
                                              f"Return present: one true/false per key fact, in order."}],
        response_format={"type": "json_schema", "json_schema": {"name": "judge", "strict": True, "schema": schema}})
    present = json.loads(resp.choices[0].message.content)["present"]
    return (present + [False] * len(key_facts))[:len(key_facts)]


def score(q: dict) -> dict:
    r = answer_question(q["question"], q["team"], qid=q["id"], verbose=False)
    present = judge(r["answer"], q["key_facts"])
    ref_turns = {(c["meeting_id"], turn_of(c["meeting_id"], c["quote"])) for c in q["citations"]}
    got_turns = {(c["meeting_id"], turn_of(c["meeting_id"], c["quote"])) for c in r["citations"]}
    near = sum(any(m == rm and t is not None and rt is not None and abs(t - rt) <= 3
                   for rm, rt in ref_turns) for m, t in got_turns)
    return {"id": q["id"], "team": q["team"], "answer": r["answer"], "citations": r["citations"],
            "key_facts": q["key_facts"], "present": present,
            "facts_found": sum(present), "facts_total": len(present),
            "valid_citations": len(r["citations"]),
            "meetings_cited": len({c["meeting_id"] for c in r["citations"]}),
            "near_reference": near, "steps": r["steps"], "cost_usd": r["cost_usd"], "seconds": r["seconds"]}


def rejudge(saved: list[dict]) -> list[dict]:
    """Re-score saved answers with the current judge, without running the agent again (cheap)."""
    for s in saved:
        s["present"] = judge(s["answer"], s["key_facts"])
        s["facts_found"] = sum(s["present"])
    return saved


def main() -> None:
    # --heldout: the 6 questions written after the code was frozen (run once; don't tune on them)
    heldout = "--heldout" in sys.argv
    qfile = "heldout_questions.json" if heldout else "my_questions.json"
    qs = json.loads((ROOT / "tests" / qfile).read_text(encoding="utf-8"))["questions"]
    out = ROOT / "tests" / f"results_{'heldout_' if heldout else ''}{MODEL}.json"
    if "--rejudge" in sys.argv:
        print(f"Re-judging saved answers in {out.name}...\n")
        results = rejudge(json.loads(out.read_text(encoding="utf-8")))
    else:
        wanted = {a for a in sys.argv[1:] if not a.startswith("--")}
        qs = [q for q in qs if not wanted or q["id"] in wanted]
        print(f"Agent {MODEL} on {len(qs)} {'held-out' if heldout else 'test'} question(s)...\n")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(score, qs))
        if wanted and out.exists():
            # Running a subset: merge into the saved results instead of overwriting them,
            # so the totals still cover all 10 questions (the other answers are kept as they were).
            kept = {s["id"]: s for s in json.loads(out.read_text(encoding="utf-8"))}
            kept.update({s["id"]: s for s in results})
            print(f"(merged {len(results)} new result(s) into {out.name}; others kept from the last run)\n")
            results = [kept[k] for k in sorted(kept)]

    for s in results:
        print(f"{s['id']} ({s['team']})  facts {s['facts_found']}/{s['facts_total']}  "
              f"citations {s['valid_citations']} valid across {s['meetings_cited']} meeting(s), "
              f"{s['near_reference']} near our key  |  {s['steps']} tool calls, ${s['cost_usd']:.4f}")
        print(f"   ANSWER: {s['answer']}")
        for fact, ok in zip(s["key_facts"], s["present"]):
            print(f"     {'+' if ok else '-'} {fact}")
        print()

    f_found = sum(s["facts_found"] for s in results)
    f_total = sum(s["facts_total"] for s in results)
    full = sum(s["facts_found"] == s["facts_total"] for s in results)
    print(f"KEY FACTS: {f_found}/{f_total} = {f_found / f_total:.0%}   fully correct answers: {full}/{len(results)}")
    print(f"CITATIONS: {sum(s['valid_citations'] for s in results)} valid; "
          f"{sum(s['meetings_cited'] >= 2 for s in results)}/{len(results)} answers cite 2+ meetings; "
          f"{sum(s['valid_citations'] == 0 for s in results)} answers with no valid citation")
    print(f"COST: ${sum(s['cost_usd'] for s in results):.3f} total, "
          f"avg {sum(s['steps'] for s in results) / len(results):.1f} tool calls, "
          f"avg {sum(s['seconds'] for s in results) / len(results):.0f}s per question")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
