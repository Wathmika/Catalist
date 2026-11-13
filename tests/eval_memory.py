"""
Phase 3 - Score the extracted memory against facts we checked by reading (plain code).

We know TS3012's story from reading the transcripts (tests/my_questions.json T01-T04
and Phase 1b). Each check below names a moment that the memory SHOULD contain:
a meeting, the turn where it's said (we allow +/- 3 turns, the model may pick a
neighbouring turn), and a word that must appear in the fact's topic or summary.

    python tests/eval_memory.py

Used to compare prompt versions and models with a number, not a feeling.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from extract import FACTS_PATH  # noqa: E402

# (description, meeting, turn, word that must appear in topic/summary)
CHECKS = [
    ("Speech recognition rejected in b (too hard)",       "TS3012b", 145, "speech"),
    ("PM wants to drop the LCD in b",                     "TS3012b", 112, "lcd"),
    ("Marketing: young people want LCD + speech",         "TS3012b", 435, "lcd"),
    ("Kinetic energy proposed by C",                      "TS3012c", 47,  "kinetic"),
    ("PM objects: remotes lie still 99% of the time",     "TS3012c", 56,  "kinetic"),
    ("Regular batteries chosen",                          "TS3012c", 75,  "batter"),
    ("LCD back in, in c",                                 "TS3012c", 168, "lcd"),
    ("Rubber chosen for the case",                        "TS3012c", 286, "rubber"),
    ("No scroll-wheel",                                   "TS3012c", 635, "scroll"),
    ("Production cost 17 EUR, over 12.5",                 "TS3012d", 32,  "17"),
    ("Speech recognition scrapped for cost",              "TS3012d", 32,  "speech"),
    ("90% want LCD vs 60% speech recognition",            "TS3012d", 38,  "lcd"),
    ("Budget is 12.5 EUR",                                "TS3012d", 48,  "12.5"),
    ("Keep rubber, lose the yellow colour",               "TS3012d", 136, "rubber"),
]
ROLES = {"A": "project manager", "B": "user interface", "C": "industrial", "D": "marketing"}


def main() -> None:
    db = sqlite3.connect(FACTS_PATH)
    model = db.execute("SELECT model, prompt_version FROM extractions WHERE meeting_id LIKE 'TS3012%' "
                       "LIMIT 1").fetchone()
    print(f"Memory for TS3012 from: {model}\n")
    hits = 0
    for desc, mid, turn, word in CHECKS:
        rows = db.execute("SELECT topic, summary, value FROM facts WHERE meeting_id = ? "
                          "AND turn BETWEEN ? AND ?", (mid, turn - 3, turn + 3)).fetchall()
        ok = any(word in f"{t} {s} {v if v is not None else ''}".lower() for t, s, v in rows)
        hits += ok
        print(f"  {'OK  ' if ok else 'MISS'} {mid} #{turn:<4} {desc}")
    roles = dict(db.execute("SELECT speaker, role FROM roles WHERE team = 'TS3012'").fetchall())
    role_ok = sum(ROLES[s] in roles.get(s, "").lower() for s in ROLES)
    print(f"\nStory checks: {hits}/{len(CHECKS)}   Roles: {role_ok}/4")
    n, tin, tout = db.execute("SELECT SUM(n_facts), SUM(tokens_in), SUM(tokens_out) FROM extractions "
                              "WHERE meeting_id LIKE 'TS3012%'").fetchone()
    print(f"Facts: {n}, tokens {tin:,} in + {tout:,} out")


if __name__ == "__main__":
    main()
