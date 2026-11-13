"""
Phase 2 - The search interface (plain code, no AI).

The agent will only ever call these functions. It never knows that SQLite is
underneath, so we can swap in pgvector or Pinecone later by changing this one
file. Every function returns small amounts of text, so the agent never sees
the whole corpus, however big it gets.

    search(query, team, meetings, k, mode)  -> best-matching chunks
        mode="keyword"  BM25 on exact words (good for names, numbers, "titanium")
        mode="vector"   meaning, via embeddings (good for "material" ~ "rubber")
        mode="hybrid"   both, merged (default; falls back to keyword if no embeddings)
    get_turns(meeting_id, start, end)       -> exact original turns
    list_meetings(team)                     -> a team's meetings in order

Try it from the Catalist folder:
    python src/search.py TS3012 "case material"            (hybrid)
    python src/search.py TS3012 "case material" keyword
"""

import re
import sqlite3
import sys

from parse import search_text
from store import DB_PATH

# Very common words carry no meaning for search ("what did they ...").
STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was
were be been being it its this that these those what which who whom whose when where why
how did do does done they them their there we our us you your he she his her i me my
so than then too very can could would should will shall may might must not no yes
about into over under again more most some any each other such only own same just
team teams first later final end""".split())


def _connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def _fts_query(query: str) -> str:
    """Turn a plain question into an FTS5 query: meaningful words joined by OR.
    BM25 then ranks chunks that contain more (and rarer) words higher."""
    words = re.findall(r"[a-z0-9.]+", search_text(query).lower())
    keep = [w.strip(".") for w in words
            if len(w.strip(".")) > 1                      # drop stray letters like "s"
            and w not in STOPWORDS
            and not re.fullmatch(r"[a-z]{2}\d{4}", w)]     # team ids like "ts3012": already a filter
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(keep))   # dedupe, keep order


def keyword_search(query: str, team: str | None = None, meetings: list[str] | None = None,
                   k: int = 8) -> list[dict]:
    """Keyword (BM25) search over chunks, optionally limited to a team or meetings.
    Filtering by team FIRST is what keeps this fast and relevant at 10,000 meetings."""
    fts = _fts_query(query)
    if not fts:
        return []
    sql = ("SELECT c.chunk_id, c.meeting_id, c.ord, c.start_turn, c.end_turn, c.text, "
           "bm25(chunks_fts) AS score "
           "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
           "WHERE chunks_fts MATCH ?")
    params: list = [fts]
    if team:
        sql += " AND c.team = ?"
        params.append(team)
    if meetings:
        sql += f" AND c.meeting_id IN ({','.join('?' * len(meetings))})"
        params += meetings
    sql += " ORDER BY score LIMIT ?"          # bm25(): lower = better match
    params.append(k)
    with _connect() as db:
        return [dict(r) for r in db.execute(sql, params)]


def vector_search(query: str, team: str | None = None, meetings: list[str] | None = None,
                  k: int = 8) -> list[dict]:
    """Meaning-based search. Embeds the query (1 small API call, cached), then
    compares it with the stored chunk vectors of THIS team only.
    OpenAI vectors have length 1, so similarity = dot product (cosine)."""
    from embed import embed_texts, open_vectors, pack, text_key, unpack

    vec_db = open_vectors()
    qkey = text_key("query:" + query)
    row = vec_db.execute("SELECT vec FROM vectors WHERE key = ?", (qkey,)).fetchone()
    if row:
        qvec = unpack(row[0]) # question vector cache, so the same question costs nothing the second time.
    else:
        qvec = embed_texts([query])[0][0]
        vec_db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (qkey, pack(qvec)))
        vec_db.commit()

    sql = "SELECT chunk_id, meeting_id, ord, start_turn, end_turn, text FROM chunks WHERE 1=1"
    params: list = []
    if team:
        sql += " AND team = ?"
        params.append(team)
    if meetings:
        sql += f" AND meeting_id IN ({','.join('?' * len(meetings))})"
        params += meetings
    with _connect() as db:
        rows = [dict(r) for r in db.execute(sql, params)]

    scored = []
    for r in rows:
        v = vec_db.execute("SELECT vec FROM vectors WHERE key = ?", (text_key(r["text"]),)).fetchone()
        if v:   # chunks not embedded yet are skipped
            r["score"] = sum(a * b for a, b in zip(qvec, unpack(v[0])))
            scored.append(r)
    scored.sort(key=lambda r: -r["score"])      # higher = more similar
    return scored[:k]


def search(query: str, team: str | None = None, meetings: list[str] | None = None,
           k: int = 8, mode: str = "hybrid") -> list[dict]:
    """The one search the agent calls. Hybrid = keyword + meaning, merged with
    Reciprocal Rank Fusion: a chunk ranked high by EITHER method rises to the top.
    RRF uses ranks, not raw scores, because BM25 and cosine scores aren't comparable."""
    if mode == "keyword":
        return keyword_search(query, team, meetings, k)
    try:
        vec_hits = vector_search(query, team, meetings, k=30 if mode == "hybrid" else k)
    except (SystemExit, Exception) as err:   # no key, no internet, no embeddings yet
        if mode == "vector":
            raise
        print(f"[search] vector search unavailable ({err.__class__.__name__}), using keyword only")
        return keyword_search(query, team, meetings, k)
    if mode == "vector":
        return vec_hits
    kw_hits = keyword_search(query, team, meetings, k=30)
    fused: dict[int, dict] = {}
    for hits in (kw_hits, vec_hits):
        for rank, h in enumerate(hits):
            entry = fused.setdefault(h["chunk_id"], {**h, "score": 0.0})
            entry["score"] += 1 / (60 + rank)          # RRF: 60 is the standard constant
    return sorted(fused.values(), key=lambda h: -h["score"])[:k]


def get_turns(meeting_id: str, start: int, end: int) -> list[dict]:
    """Exact original turns, for reading context and for quoting."""
    with _connect() as db:
        rows = db.execute("SELECT turn_idx, speaker, time, text FROM turns "
                          "WHERE meeting_id = ? AND turn_idx BETWEEN ? AND ? ORDER BY turn_idx",
                          (meeting_id, start, end))
        return [dict(r) for r in rows]


def list_meetings(team: str) -> list[dict]:
    """A team's meetings in time order. Missing ones (IS1002 has no 'a') simply aren't listed."""
    with _connect() as db:
        rows = db.execute("SELECT meeting_id, letter, ord, n_turns FROM meetings "
                          "WHERE team = ? ORDER BY ord", (team,))
        return [dict(r) for r in rows]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit()
    team, query = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    print("Meetings:", [m["meeting_id"] for m in list_meetings(team)])
    print("Mode:", mode, "| keyword query:", _fts_query(query), "\n")
    for hit in search(query, team=team, k=5, mode=mode):
        print(f"--- {hit['meeting_id']} turns #{hit['start_turn']}-#{hit['end_turn']}  "
              f"(score {hit['score']:.2f})")
        print(hit["text"][:400], "...\n")
