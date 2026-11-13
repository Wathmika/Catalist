"""
Phase 4 - Citation checking (plain code, never an LLM).

Catalist's rules: a citation is a meeting id + a quote of at least six words,
copied exactly from ONE speaker turn of that meeting. Case and spacing are
ignored; paraphrases fail. We apply the same rules before submitting, so the
grader never sees a quote we haven't already proven.

    check_citation(meeting_id, quote) -> (ok, message)
"""

import re

from search import _connect

MIN_WORDS = 6


def norm(text: str) -> str:
    """Lower-case and collapse spaces, like the grader."""
    return re.sub(r"\s+", " ", text).strip().lower()


def check_citation(meeting_id: str, quote: str) -> tuple[bool, str]:
    words = len(quote.split())
    if words < MIN_WORDS:
        return False, f"quote has {words} words; at least {MIN_WORDS} are required"
    with _connect() as db:
        rows = db.execute("SELECT turn_idx, text FROM turns WHERE meeting_id = ?", (meeting_id,)).fetchall()
    if not rows:
        return False, f"meeting {meeting_id} does not exist"
    q = norm(quote)
    for r in rows:
        if q in norm(r["text"]):
            return True, f"found in {meeting_id} turn #{r['turn_idx']}"
    # Help the agent fix it: is it in another meeting, or split across turns?
    return False, (f"these exact words are not inside any single turn of {meeting_id}. "
                   "Copy a continuous span from ONE turn exactly as shown by read_turns "
                   "(keep 'um', 'uh', repeated words; no '...').")


if __name__ == "__main__":
    print(check_citation("TS3012c", "Okay. Okay, so we're definitely going for rubber."))
    print(check_citation("TS3012c", "so we chose rubber for the remote"))
    print(check_citation("TS3012c", "going for rubber"))
