"""
Phase 1b - Small helper for reading the transcripts by hand.

Plain Python, no AI. Two uses:

1) Find turns that mention a word, in one team or one meeting:
       python src/inspect_data.py find TS3012 "scroll"
       python src/inspect_data.py find TS3012c "rubber|plastic"
   Prints: meeting, turn number, speaker, text.  (The search is a regex,
   so "a|b" means "a or b". Case is ignored.)

2) Show the turns around a specific turn, to read the context:
       python src/inspect_data.py show TS3012c 56
   Prints turns 51-61 of TS3012c.

We used this to write our own test questions (tests/my_questions.json)
with answers checked by reading the actual transcript.
"""

import re
import sys

from explore import TRANSCRIPTS_DIR, parse_turns

#check which meetings to look at
def meetings_for(prefix: str) -> list[str]:
    """'TS3012' -> all its meetings; 'TS3012c' -> just that one."""
    return sorted(p.stem for p in TRANSCRIPTS_DIR.glob(f"{prefix}*.txt"))

#search for the word by looping over each meeting, then over each turn
def find(prefix: str, pattern: str, min_words: int = 1) -> None:
    rx = re.compile(pattern, re.IGNORECASE)
    for mid in meetings_for(prefix):
        for i, t in enumerate(parse_turns(TRANSCRIPTS_DIR / f"{mid}.txt")):
            if rx.search(t["text"]) and len(t["text"].split()) >= min_words:
                print(f"{mid} #{i:<4} {t['speaker']}: {t['text'][:300]}")

#print 5 turns before and after the one we searched. So I can see the full conversation
def show(meeting_id: str, turn: int, around: int = 5) -> None:
    turns = parse_turns(TRANSCRIPTS_DIR / f"{meeting_id}.txt")
    for i in range(max(0, turn - around), min(len(turns), turn + around + 1)):
        mark = ">>" if i == turn else "  "
        print(f"{mark} #{i:<4} {turns[i]['speaker']}: {turns[i]['text']}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "find":
        find(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "show":
        show(sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
