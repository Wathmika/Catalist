"""
Phase 4 - Answer Catalist's 25 questions and write answers.json.

    python src/run_questions.py              all 25 questions
    python src/run_questions.py Q01 Q07      only these (merged into the existing answers.json)

Questions run in parallel (they're independent). Each answer is checked by code
before it is written; a full log per question goes to logs/Qxx.json.
The brief (Phase 5) is added to answers.json by src/brief.py.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from agent import MODEL, answer_question
from parse import ROOT

ANSWERS_PATH = ROOT / "answers.json"
WORKERS = 4


def main() -> None:
    data = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    wanted = set(sys.argv[1:])
    questions = [q for q in data["questions"] if not wanted or q["question_id"] in wanted]

    existing = {"answers": [], "brief": ""}
    if ANSWERS_PATH.exists():
        existing = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    by_id = {a["question_id"]: a for a in existing.get("answers", [])}

    print(f"Answering {len(questions)} question(s) with {MODEL}, {WORKERS} at a time")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda q: answer_question(q["question"], q["team"], q["question_id"]),
                                questions))

    for r in results:
        by_id[r["question_id"]] = {"question_id": r["question_id"], "answer": r["answer"],
                                   "citations": [{"meeting_id": c["meeting_id"], "quote": c["quote"]}
                                                 for c in r["citations"]]}
    ordered = [by_id[q["question_id"]] for q in data["questions"] if q["question_id"] in by_id]
    ANSWERS_PATH.write_text(json.dumps({"answers": ordered, "brief": existing.get("brief", "")},
                                       indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'ID':<4} {'cites':>5} {'steps':>5} {'cost':>8} {'secs':>5}  answer")
    for r in sorted(results, key=lambda r: r["question_id"]):
        print(f"{r['question_id']:<4} {len(r['citations']):>5} {r['steps']:>5} ${r['cost_usd']:>7.4f} "
              f"{r['seconds']:>5.0f}  {r['answer'][:70]}")
    total = sum(r["cost_usd"] for r in results)
    no_cite = [r["question_id"] for r in results if not r["citations"]]
    print(f"\nTotal ${total:.3f}, avg {sum(r['steps'] for r in results) / len(results):.1f} tool calls per question."
          f" Written to answers.json." + (f" WITHOUT valid citations: {no_cite}" if no_cite else ""))


if __name__ == "__main__":
    main()
