"""
Does memory know what is CURRENT? (plain code, no AI, no cost)

eval_memory.py checks that the extractor FOUND the right moments. This checks the next step:
after code decides what replaced what (memory.py), is the latest CURRENT settled decision on
a subject the one the team really ended with? Every expected outcome below was checked by
reading the transcripts (final meeting of each team).

    python tests/eval_current.py

Deterministic: the same memory gives the same score, so a change to memory.py can be
compared before/after without run-to-run noise.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from memory import get_memory  # noqa: E402

# (team, subject word, what the final state must say: a word in the summary, or a status)
CHECKS = [
    ("TS3012", "lcd",     "status:agreed",   "LCD kept in the final design (d)"),
    ("TS3012", "speech",  "status:rejected", "speech recognition scrapped for cost (d)"),
    ("TS3012", "scroll",  "status:rejected", "no scroll-wheel (c)"),
    ("TS3012", "rubber",  "status:agreed",   "rubber case kept over the yellow colour (d)"),
    ("IS1004", "lcd",     "status:rejected", "no LCD (dropped in c, confirmed d)"),
    ("IS1004", "speech",  "status:rejected", "no speech recognition"),
    ("IS1000", "speech",  "status:rejected", "speech recognition out (d)"),
    ("ES2012", "chip",    "standard",        "custom chip in c -> regular standard chip in d"),
    ("ES2013", "kinetic", "status:agreed",   "kinetic energy (c consensus, d confirmed)"),
    ("IS1005", "infra",   "status:agreed",   "infra-red chosen over radio waves (c)"),
    ("IS1006", "dynamo",  "status:agreed",   "hand dynamo (+ battery) in d"),
    ("ES2015", "solar",   "status:rejected", "no solar backup: fully kinetic"),
]


def final_state(team: str, word: str) -> dict | None:
    facts = [f for f in get_memory(team, topic=word)
             if f["type"] == "decision" and f["current"] and f["status"] in ("agreed", "rejected")]
    return max(facts, key=lambda f: (f["ord"], f["turn"])) if facts else None


def main() -> None:
    ok_n = 0
    for team, word, expect, desc in CHECKS:
        f = final_state(team, word)
        if f is None:
            ok, got = False, "no current settled decision found"
        elif expect.startswith("status:"):
            ok = f["status"] == expect.split(":")[1]
            got = f"{f['status']} {f['meeting_id']} #{f['turn']}: {f['summary'][:70]}"
        else:
            ok = expect in f["summary"].lower()
            got = f"{f['status']} {f['meeting_id']} #{f['turn']}: {f['summary'][:70]}"
        ok_n += ok
        print(f"  {'OK  ' if ok else 'MISS'} {team} {word:<8} {desc}\n         memory says: {got}")
    print(f"\nCurrent-state checks: {ok_n}/{len(CHECKS)}")


if __name__ == "__main__":
    main()
