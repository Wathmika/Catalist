"""
Phase 1 - Explore the data.

What this script does (plain Python, no AI, no internet):
  1. Reads every transcript in transcripts/
  2. Splits each one into speaker turns
  3. Prints numbers we need for design decisions and the cost estimate

Run from the Catalist folder:
    python src/explore.py
"""

import re
import statistics
from collections import defaultdict
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"

# Each turn in a file looks like this (header line, then one line of speech):
#   00:00:25 Speaker D
#   Yes. Um my name is Dirk ...
TURN_HEADER = re.compile(r"^(\d\d:\d\d:\d\d) Speaker ([A-D])$")

# Filler words. We only COUNT them here, we never delete them, because the
# graders check quotes word for word against the original text.
FILLERS = {"um", "uh", "hmm", "mm", "mm-hmm", "uh-huh", "yeah", "okay"}

# OpenAI models use roughly 1.3 tokens per English word. The task sheet says
# ~676k words = ~880k tokens, which matches. If the `tiktoken` package is
# installed we count exactly instead of estimating.
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    TOKEN_METHOD = "exact (tiktoken)"
except Exception:
    def count_tokens(text: str) -> int:
        return round(len(text.split()) * 1.3)
    TOKEN_METHOD = "estimated (words x 1.3)"


def parse_turns(path: Path) -> list[dict]:
    """Return a list of turns: {time, speaker, text}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    turns = []
    i = 0
    while i < len(lines):
        m = TURN_HEADER.match(lines[i].strip())
        if m and i + 1 < len(lines):
            turns.append({"time": m.group(1), "speaker": m.group(2),
                          "text": lines[i + 1].strip()})
            i += 2
        else:
            i += 1  # header block or blank line
    return turns


def main() -> None:
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    teams = defaultdict(list)          # team id -> ["a", "b", ...]
    per_meeting = []                   # one row of stats per meeting
    all_turn_lengths = []
    filler_count = 0
    total_words = 0

    for f in files:
        meeting_id = f.stem             # e.g. "TS3012a"
        team, letter = meeting_id[:-1], meeting_id[-1]
        teams[team].append(letter)

        turns = parse_turns(f)
        text = " ".join(t["text"] for t in turns)
        words = text.split()
        total_words += len(words)
        filler_count += sum(1 for w in words if w.lower().strip(".,?!") in FILLERS)
        for t in turns:
            n = len(t["text"].split())
            all_turn_lengths.append(n)

        per_meeting.append({
            "id": meeting_id,
            "turns": len(turns),
            "words": len(words),
            "tokens": count_tokens(text),
            "duration": turns[-1]["time"] if turns else "-",
        })

    total_tokens = sum(m["tokens"] for m in per_meeting)
    n = len(per_meeting)

    print("=" * 60)
    print("CORPUS OVERVIEW")
    print("=" * 60)
    print(f"Meetings:            {n}")
    print(f"Teams:               {len(teams)}")
    print(f"Total words:         {total_words:,}")
    print(f"Total tokens:        {total_tokens:,}  [{TOKEN_METHOD}]")
    print(f"Avg tokens/meeting:  {total_tokens // n:,}")
    big = max(per_meeting, key=lambda m: m["tokens"])
    small = min(per_meeting, key=lambda m: m["tokens"])
    print(f"Largest meeting:     {big['id']} ({big['tokens']:,} tokens, {big['duration']})")
    print(f"Smallest meeting:    {small['id']} ({small['tokens']:,} tokens, {small['duration']})")

    print("\n" + "=" * 60)
    print("TEAMS WITH MISSING MEETINGS")
    print("=" * 60)
    for team, letters in sorted(teams.items()):
        if sorted(letters) != ["a", "b", "c", "d"]:
            missing = sorted(set("abcd") - set(letters))
            print(f"{team}: has {''.join(sorted(letters))}, missing {''.join(missing)}")

    print("\n" + "=" * 60)
    print("SPEAKER TURNS (this decides how we chunk)")
    print("=" * 60)
    lens = sorted(all_turn_lengths)
    print(f"Total turns:          {len(lens):,}")
    print(f"Median words/turn:    {statistics.median(lens)}")
    print(f"Turns with < 6 words: {sum(1 for x in lens if x < 6) / len(lens):.0%}"
          "  (too short to be a valid citation)")
    print(f"Longest turn:         {lens[-1]} words")
    print(f"Filler words:         {filler_count / total_words:.1%} of all words")

    print("\n" + "=" * 60)
    print("WHAT THIS MEANS")
    print("=" * 60)
    print(f"Everything in one prompt:  ~{total_tokens:,} tokens -> too big, and")
    print("                           at 10,000 meetings it would be ~"
          f"{total_tokens * 10000 // n / 1e6:.0f}M tokens.")
    print(f"One team (4 meetings):     ~{total_tokens * 4 // n:,} tokens.")
    print(f"Per 1,000 meetings:        ~{total_tokens * 1000 // n / 1e6:.1f}M tokens to read once at ingest.")


if __name__ == "__main__":
    main()
