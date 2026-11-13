"""
Phase 3 - Reading the memory: timelines and "what is current" (plain code, no AI).

The LLM extracted facts in extract.py. Deciding which fact is CURRENT is not a
judgment call, it's ordering, so code does it:

  * facts are grouped by team and topic, and sorted by meeting order (a=1..d=4),
    then by turn number inside a meeting
  * a fact is SUPERSEDED if a later fact points at it ("updates"), or if a later
    decision on the same topic has a settled status (agreed / rejected)
  * the current state of a topic = its latest decision that isn't superseded

The agent calls:
    get_memory(team, topic=None, types=None, up_to_ord=None)
        up_to_ord=3 means "only what was known after meeting c" (used for the
        TS3012 brief, which must not see the final meeting)
    get_roles(team)

Try it from the Catalist folder:
    python src/memory.py TS3012
    python src/memory.py TS3012 "lcd"          (topics containing "lcd")
    python src/memory.py TS3012 --before-d     (what was known before meeting d)
"""

import re
import sqlite3
import sys

from extract import FACTS_PATH

SETTLED = {"agreed", "rejected"}


def _rows(sql: str, params: tuple) -> list[dict]:
    db = sqlite3.connect(FACTS_PATH)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, params)]
    finally:
        db.close()


_ROLE_WORDS = {"project": r"manag|project lead", "industrial": r"industrial|technic",
               "user": r"interface|user", "marketing": r"market"}
_CHAIR = r"kick-?off|agenda|welcome|minutes|first meeting|tool training|acquaintance|ready to begin"     # the project manager opens meeting a


def _role_verified(r: dict) -> bool:
    """CODE checks that the cited turn really establishes the role. extract.py only checked that the
    turn was spoken by that speaker, so 'Are we all gonna draw a cat?' (ES2013a #6) passed as
    evidence for a role. Accepted: the role word in the cited turn or the 2 turns before it
    ("Who's the interface designer?" - "That's me."), or in the same speaker's next 2 turns;
    for the project manager also the turn that opens the kick-off meeting."""
    from search import get_turns          # here, to keep memory.py importable without memory.db
    role = r["role"].lower().replace("technical", "industrial")
    key = next((k for k in _ROLE_WORDS if k in role), None)
    if key is None:
        return False
    turns = get_turns(r["meeting_id"], max(0, r["turn"] - 2), r["turn"] + 2)
    before = " ".join(t["text"].lower() for t in turns if t["turn_idx"] <= r["turn"])
    after = " ".join(t["text"].lower() for t in turns if t["turn_idx"] > r["turn"] and t["speaker"] == r["speaker"])
    own = next((t["text"].lower() for t in turns if t["turn_idx"] == r["turn"]), "")
    return bool(re.search(_ROLE_WORDS[key], before) or re.search(_ROLE_WORDS[key], after)
                or (key == "project" and re.search(_CHAIR, own)))


def get_roles(team: str) -> list[dict]:
    """Speaker letter -> role, with the turn where they said it (citable), and whether code could
    verify that turn ('verified'). Unverified roles must not be stated in answers."""
    rows = _rows("SELECT speaker, role, meeting_id, turn FROM roles WHERE team = ? ORDER BY speaker", (team,))
    for r in rows:
        r["verified"] = _role_verified(r)
    return rows


_STOP = {"the", "and", "for", "with", "remote", "control", "design", "product", "use", "type", "of", "on", "in"}


def _norm(text: str | None) -> str:
    """Lower-case and join spelled acronyms ("L_C_D_" -> "lcd"), so "lcd" finds them."""
    return re.sub(r"(?<![a-z])((?:[a-z]_)+)", lambda m: m.group(1).replace("_", ""), (text or "").lower())


def _subject_words(topic: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", _norm(topic)) if len(w) >= 3 and w not in _STOP}


def get_memory(team: str, topic: str | None = None, types: list[str] | None = None,
               up_to_ord: int | None = None) -> list[dict]:
    """A team's facts as timelines per topic, each fact marked current or superseded."""
    sql = "SELECT * FROM facts WHERE team = ?"
    params: list = [team]
    if up_to_ord:
        sql += " AND ord <= ?"
        params.append(up_to_ord)
    facts = _rows(sql + " ORDER BY ord, turn", tuple(params))
    by_id = {f["fact_id"]: f for f in facts}

    # 1) explicit links from the model: "this fact updates F12". Checked by code, because a review
    #    of all 1,765 links found 45% pointing at a different subject, 97 facts "updating" more
    #    than 3 others (one ES2012 fact "updated" 45), and proposals marking settled decisions
    #    as replaced. A link only counts if it passes all four checks below.
    superseded: set[int] = set()
    for f in facts:
        ids = [int(i) for i in str(f["updates"] or "").split(",") if i.strip().isdigit()]
        if not ids or len(ids) > 3:
            continue                                          # no link, or a scattershot link
        for i in ids:
            old = by_id.get(i)
            if old is None or (old["ord"], old["turn"]) >= (f["ord"], f["turn"]):
                continue                                      # unknown, or not earlier
            if not _subject_words(old["topic"]) & _subject_words(f["topic"]):
                continue                                      # different subject
            if old["status"] in SETTLED and f["status"] not in SETTLED:
                continue                                      # a proposal can't undo a decision
            superseded.add(i)
    # 2) implicit: a later SETTLED decision on the same topic replaces earlier decisions
    last_settled: dict[str, tuple] = {}
    for f in facts:
        if f["type"] == "decision" and f["status"] in SETTLED:
            last_settled[f["topic"]] = (f["ord"], f["turn"])
    for f in facts:
        later = last_settled.get(f["topic"])
        if f["type"] == "decision" and later and (f["ord"], f["turn"]) < later:
            superseded.add(f["fact_id"])

    for f in facts:
        f["current"] = f["fact_id"] not in superseded
    if topic:
        # Match the subject word in the topic name, the summary OR the evidence quote: topic names
        # vary ("remote control signal type" for radio vs infra-red), so topic-only missed facts.
        t = _norm(topic)
        facts = [f for f in facts
                 if t in _norm(f["topic"]) or t in _norm(f["summary"]) or t in _norm(f.get("evidence"))]
    if types:
        facts = [f for f in facts if f["type"] in types]
    return facts


def timeline_text(facts: list[dict], chronological: bool = False, evidence: bool = True) -> str:
    """Readable timeline. Grouped by topic by default; chronological=True gives one list in
    time order, used when the agent asks about a subject that the model filed under slightly
    different topic names ("lcd screen", "lcd screen size"): the latest fact is then simply last."""
    by_topic: dict[str, list] = {}
    if chronological:
        by_topic["(all matching facts, oldest first)"] = sorted(facts, key=lambda f: (f["ord"], f["turn"]))
    else:
        for f in facts:
            by_topic.setdefault(f["topic"], []).append(f)
    out = []
    for topic, items in by_topic.items():
        out.append(f"== {topic}")
        for f in items:
            mark = "CURRENT " if f["current"] else "replaced"
            val = f" [{f['value']:g} {f['unit'] or ''}]" if f["value"] is not None else ""
            who = f" who={f['who']}" if f["who"] else ""
            topic = f" [{f['topic']}]" if chronological else ""
            out.append(f"  {mark} F{f['fact_id']} {f['meeting_id']} #{f['turn']} {f['type']}/{f['status']}"
                       f"{topic}{who}{val}: {f['summary']}")
            if evidence and f.get("evidence"):
                out.append(f"           evidence: \"{f['evidence']}\"")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit()
    team = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    up_to = 3 if "--before-d" in sys.argv else None
    print("Roles:", ", ".join(f"{r['speaker']}={r['role']} ({r['meeting_id']} #{r['turn']})"
                              for r in get_roles(team)) or "none")
    facts = get_memory(team, topic=args[0] if args else None, up_to_ord=up_to)
    print(timeline_text(facts, chronological=bool(args)))