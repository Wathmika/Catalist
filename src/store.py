"""
Phase 2 - Build the memory store (plain code, no AI).

Creates data/memory.db (SQLite) with:

  meetings  one row per meeting: team, letter, order (a=1 .. d=4)
  turns     one row per speaker turn, ORIGINAL text, never modified.
            This is the source of truth for quotes.
  chunks    groups of consecutive turns (~200 words) used for searching.
            Each chunk remembers start_turn..end_turn, so a search hit can
            always be traced back to exact single turns for citing.
  chunks_fts  keyword index over the chunks (SQLite FTS5, BM25 ranking)

Run from the Catalist folder:
    python src/store.py
Re-running rebuilds everything from the transcripts (takes a few seconds).

Why SQLite: one file, no server, no account, free, and it gives us keyword
search + exact lookups + filters in one place. Swappable later (pgvector,
Pinecone) because the agent only talks to search.py.
"""

import sqlite3
import time
from pathlib import Path

from parse import ROOT, TRANSCRIPTS_DIR, all_meeting_ids, meeting_info, parse_turns, search_text

DB_PATH = ROOT / "data" / "memory.db"

CHUNK_WORDS = 200   # target size of a search chunk
OVERLAP_TURNS = 2   # last turns of a chunk repeat at the start of the next,
                    # so an exchange split across a boundary is still found whole

SCHEMA = """
CREATE TABLE meetings (
    meeting_id TEXT PRIMARY KEY,
    team       TEXT NOT NULL,
    letter     TEXT NOT NULL,
    ord        INTEGER NOT NULL,      -- 1..4, used to know what is most recent
    n_turns    INTEGER NOT NULL
);
CREATE TABLE turns (
    meeting_id TEXT NOT NULL,
    turn_idx   INTEGER NOT NULL,      -- position in the meeting, 0-based
    speaker    TEXT NOT NULL,         -- 'A'..'D'
    time       TEXT NOT NULL,
    text       TEXT NOT NULL,         -- ORIGINAL words, never cleaned
    PRIMARY KEY (meeting_id, turn_idx)
);
CREATE TABLE chunks (
    chunk_id   INTEGER PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    team       TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    start_turn INTEGER NOT NULL,
    end_turn   INTEGER NOT NULL,
    text       TEXT NOT NULL,         -- turns joined as "#12 A: ..." lines, for reading
    search     TEXT NOT NULL          -- normalised copy for the index
);
CREATE INDEX idx_chunks_team ON chunks(team, ord);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    search, content='chunks', content_rowid='chunk_id',
    tokenize='porter unicode61'       -- porter: 'batteries' also matches 'battery'
);
"""


def make_chunks(turns: list[dict]) -> list[tuple[int, int]]:
    """Group turns into (start, end) ranges of about CHUNK_WORDS words.
    A turn is never split, so every chunk maps cleanly back to whole turns."""
    ranges, start = [], 0
    while start < len(turns):
        words, end = 0, start
        while end < len(turns) and (words < CHUNK_WORDS or end == start): #long turns become a single chunk without split
            words += len(turns[end]["text"].split())
            end += 1
        ranges.append((start, end - 1))
        if end >= len(turns):
            break
        start = max(start + 1, end - OVERLAP_TURNS) #next chunk move forward, so the loop can't get stuck forever.
    return ranges


def build() -> None:
    t0 = time.time()
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)

    n_turns = n_chunks = 0
    for mid in all_meeting_ids():
        info = meeting_info(mid)
        turns = parse_turns(TRANSCRIPTS_DIR / f"{mid}.txt")
        db.execute("INSERT INTO meetings VALUES (?,?,?,?,?)",
                   (mid, info["team"], info["letter"], info["ord"], len(turns)))
        db.executemany("INSERT INTO turns VALUES (?,?,?,?,?)",
                       [(mid, i, t["speaker"], t["time"], t["text"]) for i, t in enumerate(turns)])
        for s, e in make_chunks(turns):
            body = "\n".join(f"#{i} {turns[i]['speaker']}: {turns[i]['text']}" for i in range(s, e + 1))
            db.execute("INSERT INTO chunks (meeting_id, team, ord, start_turn, end_turn, text, search) "
                       "VALUES (?,?,?,?,?,?,?)",
                       (mid, info["team"], info["ord"], s, e, body, search_text(body)))
            n_chunks += 1
        n_turns += len(turns)

    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")   # fill the index
    db.commit()
    db.close()
    print(f"Built {DB_PATH.relative_to(ROOT)}: {len(all_meeting_ids())} meetings, "
          f"{n_turns:,} turns, {n_chunks:,} chunks in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build()
