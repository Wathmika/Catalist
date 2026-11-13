"""
Phase 3 - Memory extraction: an LLM reads each meeting ONCE and writes down
the facts that matter, each tied to the exact turn it came from.

    python src/extract.py TS3012        one team (try this first)
    python src/extract.py --all         every team
    python src/extract.py TS3012 --force    redo, even if already extracted

    Model: gpt-4.1-mini by default. To compare, set EXTRACT_MODEL=gpt-5-mini in .env
    (results are stored per model, so both can be compared).

Fact types (from sorting Catalist's 25 questions in Phase 1b):
    decision      something chosen, proposed, rejected or reversed
    action_item   someone is asked / promises to do something (and later reports back)
    number        prices, costs, budget, percentages from research
    disagreement  people taking different sides
Plus roles: speaker letter -> role, from the introductions.

How the work is split (model vs code):
    LLM  : understanding messy speech ("okay so we're definitely going for rubber"
           is a decision; "maybe a rechargeable one" is only a proposal),
           and spotting that a fact updates an earlier one.
    code : everything with one right answer: is the turn number real, is the
           speaker valid, does "updates" point at a real earlier fact, storing,
           ordering, cost logging. Bad facts are dropped and counted, not trusted.

Memory over time: a team's meetings are read IN ORDER (a, b, c, d), and the
model is shown the team's facts so far. So when meeting c reverses a decision
from meeting b, the model links them ("updates": the old fact's id). This is
also how it would work in production: a new meeting is read together with
what is already known, and nothing old is re-processed.

The prompt and JSON schema live in src/prompts.py (versioned). Every fact must
carry "evidence": exact words from its turn. Code checks them: wrong turn number
-> corrected to the turn that really contains the words; words found nowhere ->
fact dropped (probably invented).

Facts live in data/facts.db (separate from memory.db, like vectors.db), so
rebuilding the search store never throws away paid extraction work.
"""

import json
import os
import re
from difflib import SequenceMatcher
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

from parse import ROOT, TRANSCRIPTS_DIR, all_meeting_ids, meeting_info, parse_turns
from prompts import EXTRACT_PROMPT_VERSION, EXTRACT_SCHEMA, EXTRACT_SYSTEM, extract_user_message

load_dotenv(ROOT / ".env")
FACTS_PATH = ROOT / "data" / "facts.db"
MODEL = os.getenv("EXTRACT_MODEL", "gpt-4.1-mini")
PROMPT_VERSION = EXTRACT_PROMPT_VERSION   # stored with every result, so runs are traceable
WORKERS = 5                    # teams processed in parallel (meetings within a team stay in order)

# USD per million tokens (input, output). Check the OpenAI pricing page.
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-5-mini": (0.25, 2.00), "gpt-4.1": (2.00, 8.00)}

# Which statuses make sense for which type (the schema can't express this, so code checks it)
# Obvious slips we can fix instead of throwing the fact away
STATUS_FIX = {("decision", "open"): "proposed", ("decision", "resolved"): "agreed",
              ("action_item", "agreed"): "open", ("disagreement", "agreed"): "resolved"}
ALLOWED_STATUS = {"decision": {"proposed", "agreed", "rejected"},
                  "action_item": {"open", "done"},
                  "disagreement": {"open", "resolved"},
                  "number": {"stated"}}

# ---------------------------------------------------------------- database

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id    INTEGER PRIMARY KEY,
    team       TEXT, meeting_id TEXT, ord INTEGER,
    type       TEXT, topic TEXT, summary TEXT, status TEXT,
    who        TEXT,              -- e.g. "A,C"
    value      REAL, unit TEXT,
    turn       INTEGER,           -- evidence turn in meeting_id
    evidence   TEXT,              -- exact words from that turn, verified by code
    updates    TEXT               -- fact_id(s) of earlier facts this one changes / follows up, e.g. "26,27"
);
CREATE TABLE IF NOT EXISTS roles (
    team TEXT, speaker TEXT, role TEXT, meeting_id TEXT, turn INTEGER,
    PRIMARY KEY (team, speaker)
);
CREATE TABLE IF NOT EXISTS extractions (   -- one row per meeting: what ran, what it cost
    meeting_id TEXT PRIMARY KEY, model TEXT, prompt_version TEXT,
    tokens_in INTEGER, tokens_out INTEGER, seconds REAL,
    n_facts INTEGER, n_dropped INTEGER, dropped TEXT,
    raw TEXT                     -- the model's raw JSON, so we can re-check without paying again
);
"""
_lock = threading.Lock()   # several threads write to the same SQLite file


def open_facts() -> sqlite3.Connection:
    FACTS_PATH.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(FACTS_PATH, check_same_thread=False)
    db.executescript(DB_SCHEMA)
    try:                                   # older facts.db from the first run: add the new column
        db.execute("ALTER TABLE extractions ADD COLUMN raw TEXT")
    except sqlite3.OperationalError:
        pass                               # column already there
    return db


# ---------------------------------------------------------------- one meeting

def format_meeting(turns: list[dict]) -> str:
    return "\n".join(f"#{i} {t['speaker']}: {t['text']}" for i, t in enumerate(turns))


def format_known(db: sqlite3.Connection, team: str, before_ord: int) -> tuple[str, dict[str, int]]:
    """The team's memory so far, shown to the model so it can link updates.
    Facts get SHORT LOCAL ids (K1, K2 ...), mapped back to database ids by code.
    Run v3 showed why: with database ids like F124 next to turn numbers like #416,
    the model sometimes wrote a turn number as a fact id ("F416")."""
    roles = db.execute("SELECT speaker, role FROM roles WHERE team = ? ORDER BY speaker",
                       (team,)).fetchall()
    facts = db.execute("SELECT fact_id, meeting_id, type, topic, status, summary FROM facts "
                       "WHERE team = ? AND ord < ? ORDER BY ord, turn", (team, before_ord)).fetchall()
    id_map = {f"K{n}": row[0] for n, row in enumerate(facts, start=1)}
    topics = sorted({row[3] for row in facts})
    lines = ["Known roles: " + (", ".join(f"{s}={r}" for s, r in roles) or "none yet"),
             "Existing topics (reuse these names): " + ("; ".join(topics) or "none yet")]
    if facts:
        lines.append("Earlier facts for this team:")
        lines += [f"K{n} [meeting {mid[-1]}] {typ} | {topic} | {status} | {summary}"
                  for n, (_, mid, typ, topic, status, summary) in enumerate(facts, start=1)]
    else:
        lines.append("Earlier facts for this team: none (first meeting).")
    return "\n".join(lines), id_map


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def call_model(system: str, user: str) -> tuple[dict, int, int]:
    """One extraction call, with a guard against a known failure: on IS1009b the model got
    stuck repeating itself until the output was cut off, leaving broken JSON. We cap the
    output length, and if the JSON is cut off or broken we retry once with a little
    randomness and a repetition penalty, which breaks the loop."""
    tokens_in = tokens_out = 0
    for attempt in (1, 2):
        extra = {"max_tokens": 12000}
        if not MODEL.startswith(("gpt-5", "o")):    # reasoning models don't accept these
            extra["temperature"] = 0 if attempt == 1 else 0.3
            if attempt == 2:
                extra["frequency_penalty"] = 0.5    # discourages repeating the same text
        resp = OpenAI().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "meeting_memory", "strict": True, "schema": EXTRACT_SCHEMA}},
            **extra,
        )
        tokens_in += resp.usage.prompt_tokens        # failed attempts still cost money: count them
        tokens_out += resp.usage.completion_tokens
        choice = resp.choices[0]
        try:
            if choice.finish_reason == "length":
                raise ValueError("output cut off at max_tokens")
            return json.loads(choice.message.content), tokens_in, tokens_out
        except ValueError as err:                    # JSONDecodeError is a ValueError
            if attempt == 2:
                raise RuntimeError(f"bad output twice ({err})") from err
            print(f"  retrying once: {err}")
    raise RuntimeError("unreachable")


def _key(word: str) -> str:
    """Compare words without case or punctuation: 'Okay,' == 'okay'."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def repair_evidence(evidence: str, turns: list[dict], turn: int) -> tuple[int, str, float] | None:
    """Models copy 'almost exactly': they drop an "uh", shorten "the the the the",
    change punctuation, or put "..." in the middle. That would fail the exact-quote
    rule, but the fact itself is usually right. So code finds the span of the REAL
    turn text that best matches, and returns those original words instead.
    Searches the claimed turn and its neighbours first, then the whole meeting.
    Returns (turn, exact original words, similarity) or None if nothing is close."""
    ev = [k for k in (_key(w) for w in evidence.replace("...", " ").split()) if k]
    if len(ev) < 4:
        return None
    near = [i for i in range(turn - 3, turn + 4) if 0 <= i < len(turns)]
    best = None
    for pool in (near, range(len(turns))):
        for i in pool:
            words = turns[i]["text"].split()
            keys = [_key(w) for w in words]
            for size in range(max(4, len(ev) - 3), len(ev) + 4):
                for start in range(0, max(1, len(words) - size + 1)):
                    window = keys[start:start + size]
                    score = SequenceMatcher(None, ev, window).ratio()
                    if best is None or score > best[2]:
                        best = (i, " ".join(words[start:start + size]), score)
        if best and best[2] >= 0.75:      # good match near the claimed turn: stop looking
            return best
    return best if best and best[2] >= 0.75 else None


def validate(result: dict, turns: list[dict], id_map: dict[str, int]) -> tuple[list, list, list]:
    """CODE checks the model's output. Returns (good facts, good roles, dropped reasons)."""
    good, roles, dropped = [], [], []
    for f in result.get("facts", []):
        if not f["topic"].strip() or not f["summary"].strip():
            dropped.append(f"empty topic/summary at #{f['turn']}")
            continue
        if f["status"] not in ALLOWED_STATUS[f["type"]]:
            fixed_status = STATUS_FIX.get((f["type"], f["status"]))
            if not fixed_status:
                dropped.append(f"status '{f['status']}' not valid for {f['type']}: {f['summary'][:60]}")
                continue
            dropped.append(f"status fixed {f['type']} {f['status']} -> {fixed_status}")
            f = {**f, "status": fixed_status}
        # Evidence check: the quoted words must really be in the stated turn.
        ev = norm(f["evidence"])
        if len(ev.split()) < 4:
            dropped.append(f"evidence too short at #{f['turn']}: {f['evidence']!r}")
            continue
        in_turn = 0 <= f["turn"] < len(turns) and ev in norm(turns[f["turn"]]["text"])
        if not in_turn:
            elsewhere = [i for i, t in enumerate(turns) if ev in norm(t["text"])]
            if elsewhere:
                fixed = min(elsewhere, key=lambda i: abs(i - f["turn"]))   # nearest exact match
                dropped.append(f"turn corrected #{f['turn']} -> #{fixed}")
                f = {**f, "turn": fixed}
            else:
                repaired = repair_evidence(f["evidence"], turns, f["turn"])
                if not repaired:
                    dropped.append(f"evidence not in transcript (#{f['turn']}): {f['evidence'][:60]!r}")
                    continue
                t_fixed, exact, score = repaired
                dropped.append(f"evidence repaired ({score:.2f}) #{f['turn']}->#{t_fixed}: "
                               f"{f['evidence'][:40]!r} -> {exact[:40]!r}")
                f = {**f, "turn": t_fixed, "evidence": exact}
        upd = None
        if f["updates"]:
            refs = re.findall(r"[A-Za-z]*\d+", str(f["updates"]))          # "K3,K7" -> ["K3", "K7"]
            real = [id_map[r.upper()] for r in refs if r.upper() in id_map]   # only ids we showed it
            if len(real) < len(refs):
                dropped.append(f"updates {f['updates']}: unknown id(s) removed")
            upd = ",".join(map(str, real)) or None
        good.append({**f, "topic": f["topic"].strip().lower(), "updates": upd})
    for r in result.get("roles", []):
        if 0 <= r["turn"] < len(turns) and turns[r["turn"]]["speaker"] == r["speaker"]:
            roles.append(r)        # a role must come from that speaker's own turn
        else:
            dropped.append(f"role {r['speaker']}={r['role']} not said by {r['speaker']} at #{r['turn']}")
    return good, roles, dropped


def extract_meeting(db: sqlite3.Connection, meeting_id: str) -> str:
    info = meeting_info(meeting_id)
    turns = parse_turns(TRANSCRIPTS_DIR / f"{meeting_id}.txt")
    with _lock:
        known, id_map = format_known(db, info["team"], info["ord"])
    user = extract_user_message(meeting_id, info["letter"], info["team"], known, format_meeting(turns))

    t0 = time.time()
    result, tin, tout = call_model(EXTRACT_SYSTEM, user)
    facts, roles, dropped = validate(result, turns, id_map)

    with _lock:
        db.execute("DELETE FROM facts WHERE meeting_id = ?", (meeting_id,))
        db.execute("DELETE FROM roles WHERE meeting_id = ?", (meeting_id,))   # no stale roles from an older run
        for f in facts:
            db.execute("INSERT INTO facts (team, meeting_id, ord, type, topic, summary, status, who, "
                       "value, unit, turn, evidence, updates) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (info["team"], meeting_id, info["ord"], f["type"], f["topic"], f["summary"],
                        f["status"], ",".join(f["who"]), f["value"], f["unit"], f["turn"],
                        f["evidence"], f["updates"]))
        for r in roles:
            db.execute("INSERT OR IGNORE INTO roles VALUES (?,?,?,?,?)",
                       (info["team"], r["speaker"], r["role"], meeting_id, r["turn"]))
        db.execute("INSERT OR REPLACE INTO extractions VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (meeting_id, MODEL, PROMPT_VERSION, tin, tout, time.time() - t0,
                    len(facts), len(dropped), json.dumps(dropped), json.dumps(result)))
        db.commit()
    return f"{meeting_id}: {len(facts)} facts, {len(roles)} roles, {len(dropped)} issues fixed or dropped, {tin}+{tout} tokens"


def extract_team(db: sqlite3.Connection, team: str, force: bool) -> None:
    """Meetings of one team strictly in order, so each sees the memory of the ones before."""
    with _lock:
        done = {r[0] for r in db.execute("SELECT meeting_id FROM extractions "
                                         "WHERE model = ? AND prompt_version = ?", (MODEL, PROMPT_VERSION))}
    for mid in [m for m in all_meeting_ids() if m[:-1] == team]:
        if mid in done and not force:
            print(f"{mid}: already extracted, skipping")
            continue
        try:
            print(extract_meeting(db, mid))
        except Exception as err:          # one failed meeting shouldn't stop the others
            print(f"{mid}: FAILED ({err.__class__.__name__}: {err})")
            return                        # later meetings need this one's memory, so stop this team


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if "--all" in sys.argv:
        teams = sorted({m[:-1] for m in all_meeting_ids()})
    elif args:
        teams = args
    else:
        print(__doc__)
        return
    if force:
        db = open_facts()
        for t in teams:      # start these teams from a clean slate
            db.execute("DELETE FROM facts WHERE team = ?", (t,))
            db.execute("DELETE FROM roles WHERE team = ?", (t,))
            db.execute("DELETE FROM extractions WHERE meeting_id LIKE ?", (t + "%",))
        db.commit()
    db = open_facts()
    t0 = time.time()
    print(f"Model {MODEL}, prompt {PROMPT_VERSION}, {len(teams)} team(s)")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda t: extract_team(db, t, force), teams))

    tin, tout, n = db.execute("SELECT SUM(tokens_in), SUM(tokens_out), COUNT(*) FROM extractions "
                              "WHERE model = ?", (MODEL,)).fetchone()
    if n:
        p_in, p_out = PRICES.get(MODEL, (0, 0))
        cost = tin / 1e6 * p_in + tout / 1e6 * p_out
        print(f"\nThis run: {time.time() - t0:.0f}s. All extractions so far with {MODEL}: "
              f"{n} meetings, {tin:,} in + {tout:,} out tokens, ${cost:.3f}")
        print(f"Per 1,000 meetings: ~${cost / n * 1000:.2f}")


if __name__ == "__main__":
    main()
