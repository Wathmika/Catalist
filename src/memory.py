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


def get_roles(team: str) -> list[dict]:
    """Speaker letter -> role, with the turn where they said it (citable)."""
    return _rows("SELECT speaker, role, meeting_id, turn FROM roles WHERE team = ? ORDER BY speaker", (team,))


def get_memory(team: str, topic: str | None = None, types: list[str] | None = None,
               up_to_ord: int | None = None) -> list[dict]:
    """A team's facts as timelines per topic, each fact marked current or superseded."""
    sql = "SELECT * FROM facts WHERE team = ?"
    params: list = [team]
    if up_to_ord:
        sql += " AND ord <= ?"
        params.append(up_to_ord)
    facts = _rows(sql + " ORDER BY ord, turn", tuple(params))

    # 1) explicit links from the model: "this fact updates F12"
    superseded = {int(i) for f in facts if f["updates"] for i in str(f["updates"]).split(",")}
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
        facts = [f for f in facts if topic.lower() in f["topic"]]
    if types:
        facts = [f for f in facts if f["type"] in types]
    return facts


def timeline_text(facts: list[dict], chronological: bool = False) -> str:
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
            if f.get("evidence"):
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
