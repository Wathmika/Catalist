"""
Phase 2 - Measure search quality on our own test questions (plain code, no AI).

For each question in tests/my_questions.json we already know which turns hold
the evidence (the turns our citations come from). We send the question text
to search() with the team filter and check: did the top-k chunks include
those turns?

    recall@k = evidence turns found in the top k chunks / all evidence turns

This is a baseline: the raw question goes in as the query. In Phase 4 the
agent writes its own, better queries and can search several times.

Run from the Catalist folder:
    python tests/eval_search.py            (compares keyword, vector, hybrid)
    python tests/eval_search.py keyword    (one mode, per-question detail)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from search import _connect, search  # noqa: E402


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def evidence_turns(citations: list[dict]) -> list[tuple[str, int]]:
    """Find which turn each test citation comes from."""
    found = []
    with _connect() as db:
        for c in citations:
            for row in db.execute("SELECT turn_idx, text FROM turns WHERE meeting_id = ?",
                                  (c["meeting_id"],)):
                if norm(c["quote"]) in norm(row["text"]):
                    found.append((c["meeting_id"], row["turn_idx"]))
                    break
    return found


def recall(question: dict, k: int, mode: str = "keyword") -> tuple[int, int]:
    hits = search(question["question"], team=question["team"], k=k, mode=mode)
    needed = evidence_turns(question["citations"])
    got = sum(any(h["meeting_id"] == m and h["start_turn"] <= t <= h["end_turn"] for h in hits)
              for m, t in needed)
    return got, len(needed)


def evaluate(questions: list[dict], mode: str, detail: bool) -> dict:
    totals = {5: [0, 0], 10: [0, 0], 20: [0, 0]}
    if detail:
        print(f"\nmode = {mode}\n{'ID':<4} {'team':<7} {'@5':>5} {'@10':>5} {'@20':>5}")
    for q in questions:
        row = []
        for k in totals:
            got, need = recall(q, k, mode)
            totals[k][0] += got
            totals[k][1] += need
            row.append(f"{got}/{need}")
        if detail:
            print(f"{q['id']:<4} {q['team']:<7} {row[0]:>5} {row[1]:>5} {row[2]:>5}")
    return {k: got / need for k, (got, need) in totals.items()}


def main() -> None:
    questions = json.loads((ROOT / "tests" / "my_questions.json").read_text(encoding="utf-8"))["questions"]
    modes = sys.argv[1:] or ["keyword", "vector", "hybrid"]
    results = {m: evaluate(questions, m, detail=len(modes) == 1) for m in modes}
    print(f"\n{'mode':<8} {'recall@5':>9} {'recall@10':>10} {'recall@20':>10}")
    for m, r in results.items():
        print(f"{m:<8} {r[5]:>9.0%} {r[10]:>10.0%} {r[20]:>10.0%}")


if __name__ == "__main__":
    main()
