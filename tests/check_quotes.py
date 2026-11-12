"""
Check that every citation in tests/my_questions.json is real.

Plain code, no AI. Applies the same rules Catalist's grader uses:
  - the quote must be at least 6 words
  - it must appear word for word inside ONE speaker turn of that meeting
  - case and spacing are ignored

Run from the Catalist folder:
    python tests/check_quotes.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from explore import TRANSCRIPTS_DIR, parse_turns  # noqa: E402


def norm(text: str) -> str:
    """Lower-case and collapse spaces, like the grader does."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    data = json.loads((ROOT / "tests" / "my_questions.json").read_text(encoding="utf-8"))
    failures = 0
    for q in data["questions"]:
        for c in q["citations"]:
            turns = parse_turns(TRANSCRIPTS_DIR / f"{c['meeting_id']}.txt")
            found = [i for i, t in enumerate(turns) if norm(c["quote"]) in norm(t["text"])]
            words = len(c["quote"].split())
            ok = bool(found) and words >= 6
            failures += not ok
            where = f"turn #{found[0]}" if found else "NOT FOUND"
            print(f"{q['id']} {c['meeting_id']:<8} {'OK  ' if ok else 'FAIL'} {words:>2} words  {where}")
    print(f"\n{len(data['questions'])} questions, {failures} failed citations")


if __name__ == "__main__":
    main()
