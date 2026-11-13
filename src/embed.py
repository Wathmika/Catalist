"""
Phase 2b - Meaning-based search: create embeddings for every chunk (uses OpenAI).

What an embedding is: a list of 1,536 numbers that represents the MEANING of
a text. Texts with similar meaning get similar numbers, even with different
words ("case material" ~ "rubber or plastic for the cover"). Keyword search
can't do that, which is why recall@10 was only 53%.

Run once from the Catalist folder (needs OPENAI_API_KEY in .env):
    python src/embed.py

Design choices:
  * Vectors live in a SEPARATE file, data/vectors.db, keyed by a hash of the
    chunk text. Rebuilding memory.db (store.py) never throws away embeddings
    we already paid for, and an unchanged chunk is never embedded twice.
  * Resumable: if it stops halfway, run it again and it continues.
  * It logs tokens and time, which gives real numbers for the
    "what does ingesting 1,000 meetings cost and how long" question.
"""

import hashlib
import os
import sqlite3
import struct
import time

from dotenv import load_dotenv
from openai import OpenAI

from parse import ROOT
from store import DB_PATH

VEC_PATH = ROOT / "data" / "vectors.db"
MODEL = "text-embedding-3-small"   # cheapest OpenAI embedding model, 1,536 numbers
PRICE_PER_M_TOKENS = 0.02          # USD per million tokens (check OpenAI pricing page)
BATCH = 100                        # chunks per API call (fewer calls = faster)


# No need to pay twice for the same chunk. If the embd model change, new keys will be stored for each model.
def text_key(text: str) -> str:
    """Same text -> same key, so we never pay to embed identical text twice."""
    return hashlib.sha1(f"{MODEL}|{text}".encode("utf-8")).hexdigest()

# pack 1536 numbers into 4 bytes when saving (SQLite cannot store python list directly)
def pack(vec: list[float]) -> bytes:
    """List of floats -> compact bytes for storing in SQLite (4 bytes per number)."""
    return struct.pack(f"{len(vec)}f", *vec)

# unpack back to number when searching
def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))

# opening the vector database.
def open_vectors() -> sqlite3.Connection:
    VEC_PATH.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(VEC_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    return db

#connecting to openAI
def client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("No OPENAI_API_KEY found. Put it in the .env file in the Catalist folder.")
    return OpenAI()

# sends the list of text and gets back one vector per text with token usage.
def embed_texts(texts: list[str]) -> tuple[list[list[float]], int]:
    """One API call for many texts. Returns (vectors, tokens used)."""
    resp = client().embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in resp.data], resp.usage.total_tokens


def main() -> None:
    mem = sqlite3.connect(DB_PATH)
    chunks = mem.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    vec_db = open_vectors()
    have = {row[0] for row in vec_db.execute("SELECT key FROM vectors")}
    todo = [(cid, text) for cid, text in chunks if text_key(text) not in have]
    print(f"{len(chunks):,} chunks, {len(chunks) - len(todo):,} already embedded, {len(todo):,} to do")

    t0, tokens = time.time(), 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vectors, used = embed_texts([text for _, text in batch])
        vec_db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                           [(text_key(text), pack(v)) for (_, text), v in zip(batch, vectors)])
        vec_db.commit()                 # save after every batch, so a crash loses little
        tokens += used
        print(f"  {min(i + BATCH, len(todo)):>5}/{len(todo)}  tokens so far {tokens:,}")

    secs = time.time() - t0
    cost = tokens / 1e6 * PRICE_PER_M_TOKENS
    print(f"\nDone in {secs:.0f}s. Tokens: {tokens:,}. Cost: ${cost:.4f}")
    if todo:
        meetings = mem.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        print(f"Per 1,000 meetings: ~{tokens / meetings * 1000 / 1e6:.1f}M tokens, "
              f"~${cost / meetings * 1000:.2f}, ~{secs / meetings * 1000 / 60:.0f} min "
              f"(one process; parallel calls would be much faster)")


if __name__ == "__main__":
    main()
