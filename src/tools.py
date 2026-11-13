"""
Phase 4 - The agent's tools (plain code, no AI).

Each tool does one lookup and returns a SMALL piece of text, so the agent's
context stays small however many meetings exist. The agent (LLM) only chooses
which tool to call with which arguments; everything below is deterministic.

    list_meetings(team)                          which meetings exist, in order
    get_roles(team)                              speaker letter -> role, with the intro turn
    get_memory(team, topic, up_to_meeting)       extracted facts as a dated timeline
    search(team, query, meetings)                hybrid keyword + meaning search over chunks
    read_turns(meeting_id, start_turn, end_turn) exact original words, for quoting

Guardrail (code, not prompt): every tool is locked to the team named in the
question. If the agent asks about another team, the tool refuses.
"""

import json

from memory import get_memory as _get_memory, get_roles as _get_roles, timeline_text
from search import get_turns, list_meetings as _list_meetings, search as _search

MAX_CHARS = 12000          # cap on any single tool result (~3k tokens)
MAX_TURNS = 30             # cap on read_turns per call


def _cap(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... cut: result too long. Narrow it (topic / meetings / turn range).]"


def _ord(letter: str | None) -> int | None:
    return "abcd".index(letter.lower()) + 1 if letter and letter.lower() in "abcd" else None


# ---------------------------------------------------------------- the tools

def list_meetings(team: str) -> str:
    ms = _list_meetings(team)
    if not ms:
        return f"No meetings found for team {team}."
    return "\n".join(f"{m['meeting_id']} (meeting {m['letter']}, {m['n_turns']} turns)" for m in ms)


ALL_ROLES = ["Project Manager", "Industrial Designer", "User Interface Designer", "Marketing Expert"]


def get_roles(team: str) -> str:
    """Roles the speakers stated themselves, VERIFIED by code (the cited turn must really state the
    role; see memory._role_verified). If exactly one speaker and one role are left after the
    verified ones, code fills the gap by elimination and says so.
    Found in testing: with D missing for TS3008, the agent guessed D's role and got it wrong; and on
    the held-out set an unverified role (ES2013 B, from a turn about drawing an animal) was wrong."""
    roles = [r for r in _get_roles(team) if r["verified"]]
    lines = [f"Speaker {r['speaker']} = {r['role']} (said in {r['meeting_id']} turn #{r['turn']})" for r in roles]
    known_speakers = {r["speaker"] for r in roles}
    known_roles = {next((x for x in ALL_ROLES if x.split()[0].lower() in r["role"].lower()), r["role"])
                   for r in roles}
    missing_s = [s for s in "ABCD" if s not in known_speakers]
    missing_r = [x for x in ALL_ROLES if x not in known_roles]
    if len(missing_s) == 1 and len(missing_r) == 1 and len(roles) == 3:
        lines.append(f"Speaker {missing_s[0]} = {missing_r[0]} (deduced: the only role left; not stated directly)")
    elif missing_s:
        lines.append(f"Roles of {', '.join(missing_s)} not verified: give only the letter.")
    return "\n".join(lines)


def get_memory(team: str, topic: str | None = None, up_to_meeting: str | None = None) -> str:
    facts = _get_memory(team, topic=topic, up_to_ord=_ord(up_to_meeting))
    if not facts:
        hint = f" about '{topic}'" if topic else ""
        return f"No extracted facts{hint} for {team}. Try another topic word, or use search."
    if topic:
        text = timeline_text(facts, chronological=True)
        head = "Facts about '%s', oldest first (CURRENT = latest on that topic):\n" % topic
    else:
        # Whole timeline: compact (no evidence quotes) so a full team fits; fetch a topic for quotes.
        text = timeline_text(facts, evidence=False)
        head = ("All facts, grouped by topic (CURRENT = latest on that topic). Evidence quotes are left out "
                "here: call get_memory with a topic, or read_turns, to get exact words:\n")
    return _cap(head + text, limit=20000 if not topic else MAX_CHARS)


def search(team: str, query: str, meetings: list[str] | None = None, k: int = 6,
           chars: int = 1800) -> str:
    hits = _search(query, team=team, meetings=meetings or None, k=k)
    if not hits:
        return "No matching passages."
    hits.sort(key=lambda h: (h["ord"], h["start_turn"]))     # show in time order
    parts = [f"--- {h['meeting_id']} turns #{h['start_turn']}-#{h['end_turn']}\n{h['text'][:chars]}"
             for h in hits]
    return _cap("\n".join(parts))


def read_turns(meeting_id: str, start_turn: int, end_turn: int) -> str:
    end_turn = min(end_turn, start_turn + MAX_TURNS - 1)
    turns = get_turns(meeting_id, start_turn, end_turn)
    if not turns:
        return f"No turns {start_turn}-{end_turn} in {meeting_id}."
    return _cap("\n".join(f"#{t['turn_idx']} {t['speaker']}: {t['text']}" for t in turns))


# ---------------------------------------------------------------- tool specs for the API

_TEAM = {"type": "string", "description": "Team id, e.g. TS3012"}
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_meetings", "description": "List the team's meetings in time order (some teams lack a or d).",
        "parameters": {"type": "object", "properties": {"team": _TEAM}, "required": ["team"]}}},
    {"type": "function", "function": {
        "name": "get_roles", "description": "Speaker letter -> role for the team, with the turn where they said it.",
        "parameters": {"type": "object", "properties": {"team": _TEAM}, "required": ["team"]}}},
    {"type": "function", "function": {
        "name": "get_memory",
        "description": ("Timeline of facts extracted from the team's meetings (decisions, action items, numbers, "
                        "disagreements), oldest first, each with meeting, turn, status and an exact evidence "
                        "quote. Use a topic word (e.g. 'lcd', 'material', 'price', 'battery') to focus; "
                        "omit it for everything. up_to_meeting='c' hides later meetings."),
        "parameters": {"type": "object", "properties": {
            "team": _TEAM,
            "topic": {"type": "string", "description": "Word that appears in the topic name, e.g. 'lcd'"},
            "up_to_meeting": {"type": "string", "enum": ["a", "b", "c", "d"]}},
            "required": ["team"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": ("Search the team's transcripts (keyword + meaning). Returns passages with turn numbers. "
                        "Use short focused queries, one per sub-question; numbers can be words or digits."),
        "parameters": {"type": "object", "properties": {
            "team": _TEAM,
            "query": {"type": "string"},
            "meetings": {"type": "array", "items": {"type": "string"},
                         "description": "Optional: limit to these meeting ids, e.g. ['TS3012c']"}},
            "required": ["team", "query"]}}},
    {"type": "function", "function": {
        "name": "read_turns",
        "description": "Exact original turns (max 30). Use this to read context and to copy quotes exactly.",
        "parameters": {"type": "object", "properties": {
            "meeting_id": {"type": "string"},
            "start_turn": {"type": "integer"}, "end_turn": {"type": "integer"}},
            "required": ["meeting_id", "start_turn", "end_turn"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer with 1-3 citations. Quotes are checked word for word.",
        "parameters": {"type": "object", "properties": {
            "parts": {"type": "array", "description": (
                "Checklist: every part the question asks (e.g. 'who proposed', 'what was chosen in the end', "
                "'why'), each with the meeting + turn that supports your answer to it, or 'not found'."),
                "items": {"type": "object", "properties": {
                    "part": {"type": "string"},
                    "evidence": {"type": "string", "description": "e.g. 'TS3012c #286', or 'not found'"}},
                    "required": ["part", "evidence"]}},
            "answer": {"type": "string", "description": "1-3 plain sentences."},
            "citations": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object", "properties": {
                    "meeting_id": {"type": "string"},
                    "quote": {"type": "string",
                              "description": "6+ consecutive words copied exactly from ONE turn"}},
                "required": ["meeting_id", "quote"]}}},
            "required": ["parts", "answer", "citations"]}}},
]

_FUNCS = {"list_meetings": list_meetings, "get_roles": get_roles, "get_memory": get_memory,
          "search": search, "read_turns": read_turns}


def run_tool(name: str, args: dict, team: str, up_to_meeting: str | None = None) -> str:
    """Run one tool call from the agent. Code enforces the guardrails:
    the agent can only look at the question's team, and (for the brief) not past a cutoff meeting."""
    if name not in _FUNCS:
        return f"Unknown tool {name}."
    if "team" in args and args["team"] != team:
        return f"Refused: this question is about team {team}, not {args['team']}."
    if name == "read_turns" and not str(args.get("meeting_id", "")).startswith(team):
        return f"Refused: only meetings of team {team} can be read."
    if up_to_meeting:                       # brief mode: hide the future
        cutoff = _ord(up_to_meeting)
        if name == "get_memory":
            args["up_to_meeting"] = up_to_meeting
        if name == "search":
            allowed = [m["meeting_id"] for m in _list_meetings(team) if m["ord"] <= cutoff]
            args["meetings"] = [m for m in (args.get("meetings") or allowed) if m in allowed] or allowed
        if name == "read_turns" and _ord(args["meeting_id"][-1]) > cutoff:
            return f"Refused: meetings after {team}{up_to_meeting} are not available for this task."
        if name == "list_meetings":
            return "\n".join(line for line in list_meetings(team).splitlines()
                             if _ord(line.split()[0][-1]) <= cutoff)
    try:
        return _FUNCS[name](**args)
    except TypeError as err:                 # wrong / missing arguments from the model
        return f"Bad arguments for {name}: {err}"


if __name__ == "__main__":
    print(run_tool("list_meetings", {"team": "TS3012"}, "TS3012"))
    print(run_tool("get_roles", {"team": "TS3012"}, "TS3012"))
    print(run_tool("get_memory", {"team": "TS3012", "topic": "lcd"}, "TS3012")[:800])
    print(run_tool("read_turns", {"meeting_id": "TS3012c", "start_turn": 55, "end_turn": 57}, "TS3012"))
    print(run_tool("search", {"team": "IS1004", "query": "x"}, "TS3012"))
    print(json.dumps(TOOL_SPECS)[:100])