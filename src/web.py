"""
A small local demo page for the agent (standard library only - nothing to install).

    python src/web.py            then open http://127.0.0.1:8000

- Ask: pick a team, type a question, and watch each step of the agent appear as it runs
  (plan -> memory / search / read -> submit -> code checks), then the answer with its citations,
  tool calls, tokens, cost and time. Every run is also logged to logs/web-<time>.json.
- Brief: the TS3012 brief from brief_TS3012.md (written by src/brief.py).

It only listens on this computer (127.0.0.1). The API key stays in .env and never reaches the page.
This is a viewer for the agent, not a product: no accounts, no chat history.
"""

import json
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import MODEL, answer_question
from parse import ROOT
from store import DB_PATH

HOST, PORT = "127.0.0.1", 8000
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def teams() -> list[dict]:
    db = sqlite3.connect(DB_PATH)
    rows = db.execute("SELECT team, GROUP_CONCAT(letter, '') FROM meetings GROUP BY team ORDER BY team").fetchall()
    db.close()
    return [{"team": t, "meetings": "".join(sorted(m))} for t, m in rows]


def run_job(job_id: str, team: str, question: str) -> None:
    job = JOBS[job_id]

    def on_step(line: str) -> None:
        with LOCK:
            job["steps"].append({"t": round(time.time() - job["start"], 1), "line": line})

    try:
        r = answer_question(question, team, qid=f"web-{time.strftime('%H%M%S')}", verbose=True, on_step=on_step)
        result = {k: r[k] for k in ("answer", "citations", "steps", "tokens_in", "tokens_out",
                                    "cost_usd", "seconds", "model", "dropped_citations")}
        with LOCK:
            job["result"] = result
    except Exception as e:                       # show the error on the page instead of hanging
        with LOCK:
            job["error"] = f"{e.__class__.__name__}: {e}"
    finally:
        with LOCK:
            job["done"] = True


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, status=200, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):                # keep the terminal for the agent's own steps
        pass

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/":
            return self._send(PAGE.replace("__MODEL__", MODEL), ctype="text/html; charset=utf-8")
        if path == "/api/teams":
            return self._send(teams())
        if path == "/api/brief":
            f = ROOT / "brief_TS3012.md"
            return self._send({"text": f.read_text(encoding="utf-8") if f.exists() else ""})
        if path == "/api/job":
            job_id = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("id", "")
            with LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return self._send({"error": "unknown job"}, 404)
                return self._send({k: job.get(k) for k in ("steps", "done", "result", "error")})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/ask":
            return self._send({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        team, question = str(body.get("team", "")).strip(), str(body.get("question", "")).strip()
        if team not in {t["team"] for t in teams()} or not question:
            return self._send({"error": "pick a team and type a question"}, 400)
        job_id = uuid.uuid4().hex[:8]
        with LOCK:
            JOBS[job_id] = {"steps": [], "done": False, "result": None, "error": None, "start": time.time()}
        threading.Thread(target=run_job, args=(job_id, team, question[:1000]), daemon=True).start()
        self._send({"id": job_id})


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meeting memory agent</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#667085; --line:#e4e7ec; --accent:#2f6fdf;
          --ok:#1a7f4b; --warn:#b54708; --bad:#c0362c; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:20px 28px 0; max-width:1180px; margin:auto; }
  h1 { font-size:20px; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:13px; }
  nav { display:flex; gap:4px; margin-top:16px; border-bottom:1px solid var(--line); }
  nav button { border:0; background:none; padding:10px 14px; font:inherit; color:var(--muted); cursor:pointer; border-bottom:2px solid transparent; }
  nav button.on { color:var(--ink); border-color:var(--accent); font-weight:600; }
  main { max-width:1180px; margin:auto; padding:20px 28px 40px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; }
  @media (max-width: 900px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:0 0 12px; }
  label { font-size:13px; color:var(--muted); display:block; margin-bottom:4px; }
  select, textarea { width:100%; font:inherit; padding:9px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }
  textarea { min-height:84px; resize:vertical; }
  .row { display:flex; gap:12px; align-items:end; margin-bottom:12px; }
  .row > div:first-child { width:170px; flex:none; }
  .row > div:last-child { flex:1; }
  .ex { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 14px; }
  .ex button { font:12px/1.3 inherit; border:1px solid var(--line); background:#fafbfc; border-radius:14px; padding:5px 10px; cursor:pointer; color:var(--ink); }
  .ex button:hover { border-color:var(--accent); }
  .go { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 18px; font:inherit; font-weight:600; cursor:pointer; }
  .go:disabled { opacity:.5; cursor:default; }
  .steps { list-style:none; margin:0; padding:0; max-height:520px; overflow:auto; }
  .steps li { display:grid; grid-template-columns: 44px 86px 1fr; gap:8px; padding:7px 0; border-bottom:1px solid #f0f2f5; font-size:13px; align-items:start; }
  .t { color:var(--muted); font-variant-numeric:tabular-nums; }
  .tag { font-size:11px; font-weight:700; letter-spacing:.03em; padding:2px 6px; border-radius:5px; text-align:center; background:#eef2f8; color:#344054; }
  .tag.plan { background:#e8f0fe; color:#1f4fb3; } .tag.memory { background:#f1eafd; color:#5b2fb3; }
  .tag.search { background:#e7f6ef; color:#16683d; } .tag.read { background:#fdf3e4; color:#93530c; }
  .tag.submit { background:#e6f4ea; color:var(--ok); } .tag.check { background:#fdecea; color:var(--bad); }
  .tag.outcome { background:#fff4e5; color:var(--warn); }
  .line { word-break:break-word; white-space:pre-line; }
  .empty { color:var(--muted); font-size:14px; }
  .answer { font-size:16px; line-height:1.6; margin:0 0 14px; }
  .cite { border-left:3px solid var(--accent); background:#f8faff; padding:8px 12px; margin:8px 0; border-radius:0 8px 8px 0; font-size:14px; }
  .cite b { font-size:12px; color:var(--accent); margin-right:6px; }
  .stats { display:flex; flex-wrap:wrap; gap:18px; margin-top:14px; padding-top:12px; border-top:1px solid var(--line); font-size:13px; color:var(--muted); }
  .stats b { color:var(--ink); font-size:15px; display:block; }
  .err { color:var(--bad); }
  .spin { display:inline-block; width:12px; height:12px; border:2px solid var(--line); border-top-color:var(--accent); border-radius:50%; animation:s 1s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes s { to { transform:rotate(360deg); } }
  .brief h3 { font-size:15px; margin:22px 0 8px; }
  .brief li { margin:8px 0; }
  .brief .q { display:block; font-size:12.5px; color:var(--muted); margin-top:3px; }
  .note { font-size:13px; color:var(--muted); background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:12px; }
</style></head>
<body>
<header>
  <h1>Meeting memory agent</h1>
  <div class="sub">138 AMI meetings &middot; 35 teams &middot; agent model <b>__MODEL__</b> &middot; every quote checked by code</div>
  <nav><button id="tabAsk" class="on">Ask</button><button id="tabBrief">Brief for TS3012</button></nav>
</header>
<main>
  <section id="ask">
    <div class="card" style="margin-bottom:20px">
      <div class="row">
        <div><label for="team">Team</label><select id="team"></select></div>
        <div><label for="q">Question</label><textarea id="q" placeholder="e.g. What did they first choose, what did they end with, and why?"></textarea></div>
      </div>
      <div class="ex" id="ex"></div>
      <button class="go" id="go">Ask</button> <span id="status" class="sub"></span>
    </div>
    <div class="grid">
      <div class="card"><h2>Agent steps (live)</h2><ul class="steps" id="steps"><li class="empty" style="display:block">Steps appear here while the agent works.</li></ul></div>
      <div class="card"><h2>Answer</h2><div id="out"><p class="empty">The answer and its citations appear here.</p></div></div>
    </div>
  </section>
  <section id="brief" hidden>
    <div class="card brief">
      <div class="note">Written by <code>src/brief.py</code> from meetings a&ndash;c only. The cutoff is enforced in code: the tools can't see meeting d, and any citation from d is rejected.</div>
      <div id="briefBody" class="empty">Loading&hellip;</div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const EXAMPLES = [
  ["IS1004", "Why did the team decide against speech recognition, and what did the final remote have instead?"],
  ["TS3012", "What case material did Team TS3012 choose, why, and what did they give up to stay in budget?"],
  ["ES2012", "In meeting c Team ES2012 agreed on a custom chip. What chip did they end up with?"],
  ["TS3008", "Who disagreed with aiming the remote at people under forty, and what was decided?"],
];
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

function tagFor(line) {
  if (/code check/.test(line)) return ["CHECK", "check"];
  if (/outcome check/.test(line)) return ["MEMORY CHECK", "outcome"];
  const m = line.match(/step \d+: (\w+)/); const n = m ? m[1] : "";
  return ({plan_searches:["PLAN","plan"], get_memory:["MEMORY","memory"], search:["SEARCH","search"],
           read_turns:["READ","read"], get_roles:["ROLES",""], list_meetings:["MEETINGS",""],
           submit_answer:["SUBMIT","submit"]})[n] || ["STEP",""];
}

async function loadTeams() {
  const ts = await (await fetch("/api/teams")).json();
  for (const t of ts) { const o = el("option", "", `${t.team} (${t.meetings})`); o.value = t.team; $("team").append(o); }
  $("team").value = "IS1004";
  for (const [team, q] of EXAMPLES) {
    const b = el("button", "", `${team}: ${q.length > 58 ? q.slice(0, 58) + "…" : q}`);
    b.onclick = () => { $("team").value = team; $("q").value = q; };
    $("ex").append(b);
  }
}

let timer = null;
async function ask() {
  const team = $("team").value, question = $("q").value.trim();
  if (!question) { $("q").focus(); return; }
  $("go").disabled = true; $("status").innerHTML = '<span class="spin"></span>working…';
  $("steps").innerHTML = ""; $("out").innerHTML = '<p class="empty">Waiting for the agent…</p>';
  const r = await fetch("/api/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({team, question})});
  const j = await r.json();
  if (!r.ok) { $("status").textContent = j.error; $("go").disabled = false; return; }
  let shown = 0;
  timer = setInterval(async () => {
    const s = await (await fetch("/api/job?id=" + j.id)).json();
    for (; shown < s.steps.length; shown++) {
      const st = s.steps[shown], [label, cls] = tagFor(st.line), li = el("li");
      li.append(el("span", "t", st.t.toFixed(1) + "s"), el("span", "tag " + cls, label),
                el("span", "line", st.line.replace(/^step \d+: /, "").split(" | ").join("\n• ")));
      $("steps").append(li); $("steps").scrollTop = $("steps").scrollHeight;
    }
    if (s.done) { clearInterval(timer); $("go").disabled = false; $("status").textContent = ""; show(s); }
  }, 400);
}

function show(s) {
  const out = $("out"); out.innerHTML = "";
  if (s.error) { out.append(el("p", "err", s.error)); return; }
  const r = s.result;
  out.append(el("p", "answer", r.answer || "(no answer)"));
  for (const c of r.citations) {
    const d = el("div", "cite"); d.append(el("b", "", c.meeting_id + (c.turn !== undefined ? " #" + c.turn : "")), document.createTextNode("“" + c.quote + "”"));
    out.append(d);
  }
  const st = el("div", "stats");
  for (const [k, v] of [["tool calls", r.steps], ["tokens", (r.tokens_in + r.tokens_out).toLocaleString()],
                        ["cost", "$" + r.cost_usd.toFixed(4)], ["time", r.seconds + "s"], ["model", r.model],
                        ["citations checked", r.citations.length + " valid"]]) {
    const d = el("div"); d.append(el("b", "", String(v)), document.createTextNode(k)); st.append(d);
  }
  out.append(st);
}

async function loadBrief() {
  const {text} = await (await fetch("/api/brief")).json();
  const body = $("briefBody"); body.innerHTML = ""; body.className = "";
  if (!text) { body.textContent = "No brief yet - run: python src/brief.py"; return; }
  let ul = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim(); if (!line) continue;
    if (line.startsWith("- ")) {
      if (!ul) { ul = el("ul"); body.append(ul); }
      const m = line.slice(2).match(/^(.*?)\s*(\[TS3012.*\])\s*$/), li = el("li");
      li.append(document.createTextNode(m ? m[1] : line.slice(2)));
      if (m) li.append(el("span", "q", m[2]));
      ul.append(li);
    } else { ul = null; body.append(el(line.endsWith(":") ? "h3" : "p", "", line)); }
  }
}

$("go").onclick = ask;
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); });
$("tabAsk").onclick = () => { $("ask").hidden = false; $("brief").hidden = true; $("tabAsk").className = "on"; $("tabBrief").className = ""; };
$("tabBrief").onclick = () => { $("ask").hidden = true; $("brief").hidden = false; $("tabBrief").className = "on"; $("tabAsk").className = ""; loadBrief(); };
loadTeams();
</script>
</body></html>
"""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Demo page: http://{HOST}:{PORT}   (agent model {MODEL}; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
