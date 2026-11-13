"""
Phase 2 - Parsing (plain code, no AI).

Everything that reads a transcript goes through this file, so there is one
parser and one set of rules.

Two kinds of text:
  * ORIGINAL text  -> stored as-is, used for quotes (graders check word for word)
  * SEARCH text    -> a normalised copy used only for searching, so that
                      "L_C_D_" matches "lcd" and "twenty five" matches "25"
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"

# spot the new turn starts and pulls the time and speaker letter.
TURN_HEADER = re.compile(r"^(\d\d:\d\d:\d\d) Speaker ([A-D])$")


# Identify the meeting to get facts using the file name.

def meeting_info(meeting_id: str) -> dict:
    """'TS3012c' -> {'meeting_id': 'TS3012c', 'team': 'TS3012', 'letter': 'c', 'ord': 3}.
    The order comes from the file name: a=1, b=2, c=3, d=4. This is how we
    know which decision came later (memory over time)."""
    letter = meeting_id[-1]
    return {"meeting_id": meeting_id, "team": meeting_id[:-1],
            "letter": letter, "ord": "abcd".index(letter) + 1} # +1 BECAUSE counting from 0

# list all meeting names from the file name
def all_meeting_ids() -> list[str]:
    return sorted(p.stem for p in TRANSCRIPTS_DIR.glob("*.txt"))


def parse_turns(path: Path) -> list[dict]:
    """Transcript file -> list of {time, speaker, text}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    turns, i = [], 0
    while i < len(lines):
        m = TURN_HEADER.match(lines[i].strip())
        if m and i + 1 < len(lines):
            turns.append({"time": m.group(1), "speaker": m.group(2),
                          "text": lines[i + 1].strip()})
            i += 2
        else:
            i += 1
    return turns


# Cleanup for search

# Spelled-out acronyms: "L_C_D_" -> "LCD", "T_V_" -> "TV"
_ACRONYM = re.compile(r"\b((?:[A-Za-z]_){2,})")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMBER_WORD = set(_UNITS) | set(_TENS) | {"hundred", "point", "and", "a", "half"}


def _words_to_number(words: list[str]) -> float | None:
    """['twenty','five'] -> 25, ['fourteen','point','six'] -> 14.6,
    ['twelve','and','a','half'] -> 12.5. Returns None if it isn't a number."""
    total, current, decimals, i = 0, 0, None, 0
    while i < len(words):
        w = words[i]
        if w in _UNITS:
            if decimals is not None:
                decimals += str(_UNITS[w])
            else:
                current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w == "point":
            decimals = ""
        elif words[i:i + 3] == ["and", "a", "half"]:
            return total + current + 0.5
        else:
            return None
        i += 1
    value = total + current
    if decimals:
        value = float(f"{value}.{decimals}")
    return value


def numbers_in(text: str) -> list[str]:
    """Find spoken numbers and return them as digit strings: 'twenty five Euros' -> ['25']."""
    tokens = re.findall(r"[a-z]+", text.lower())
    found, run = [], []
    for tok in tokens + ["<end>"]:
        if tok in _NUMBER_WORD:
            run.append(tok)
            continue
        while run and run[-1] in {"and", "a", "point"}:   # "five and" -> "five"
            run.pop()
        if run and run[0] not in {"and", "a", "point", "half"}:
            value = _words_to_number(run)
            if value is not None:
                found.append(f"{value:g}")
        run = []
    return found


def search_text(text: str) -> str:
    """Normalised copy for the search index ONLY. Original text is never changed."""
    t = _ACRONYM.sub(lambda m: m.group(1).replace("_", ""), text)
    nums = numbers_in(text)
    return t + (" " + " ".join(nums) if nums else "")


if __name__ == "__main__":
    # Quick self-check: python src/parse.py
    examples = ["It's gonna cost us fourteen point six Euros",
                "remember it has to be under twelve and a half",
                "the selling price is twenty five Euros",
                "I think we should lose the L_C_D_ screen"]
    for e in examples:
        print(f"{e!r}\n   -> {search_text(e)!r}")
