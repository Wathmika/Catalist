# Meeting memory agent: write-up

Plain Python, the OpenAI SDK and SQLite. Diagram: `architecture.drawio`. Every measurement and decision: `NOTES.md`.

## Architecture
- **Ingest, once per meeting:** split each transcript into speaker turns (regex). Store the turns plus ~200-word
  chunks with a keyword index (SQLite FTS5), embed the chunks (text-embedding-3-small), and have gpt-4.1-mini
  extract decisions, actions, numbers, disagreements and roles. Meetings are read in order a→d, each together
  with the team's earlier facts.
- **Answer:** a hand-written tool-calling loop (gpt-6-luna, max 12 steps). It plans one search per part of the
  question, then uses hybrid search (keyword + vectors), reads exact turns, and looks up the memory timeline and
  verified roles. Every submitted answer is checked by code and sent back with reasons if it fails.
- **Scale:** every tool is filtered by team, so a question costs the same at 138 or 10,000 meetings.
  Ingest grows linearly. The brief uses the same loop; code blocks the final meeting.

## Model vs code
| LLM (judgment) | Code (exact, cheap, repeatable) |
|---|---|
| Proposal vs decision, who owns what | Parsing, chunking, indexing, number words → digits |
| What to look up next; writing the answer | Every quote checked word for word (≥6 words, one turn) |
| Choosing the brief's points | What is *current* (latest settled decision); roles verified against their evidence turn |
| | Team lock, brief cutoff, 1–3 sentences, stripping markers |

**Rule:** when a prompt rule only worked some of the time, I moved the check into code.

## Proof
- **My own test questions** (answers written by reading; 48 citations verified by code): 10 development
  questions and 6 held-out questions on teams no other set uses. Key facts found, by my reading: **~79% with
  gpt-6-luna vs ~72% with gpt-4.1-mini**. Every answer has valid citations.
- The LLM judge was unreliable (5 wrong calls in one run), so every score was checked by hand.
- **Memory test** (does memory know the final state?): 6/12 before the memory fixes, 11/12 after.
- **Cost:** ingest **$1.06 for 138 meetings → ~$7.70 and ~80 min per 1,000** (5 workers; about half with the
  Batch API). Answers: **~$0.005 and ~13 s per question**.

## Three questions it gets wrong
1. **H04:** it misses that the custom chip became a standard chip. That change *is* in memory, but the agent
   stopped without asking memory. **Fix:** give the model the subject's timeline in the plan step.
2. **T02:** it names one objector and misses the project manager's main objection. It took part of the
   answer for the whole. **Fix:** a per-part evidence checklist.
3. **Brief:** it lists a research task as open, although meeting c had ruled out the rechargeable remote.
   **Fix:** "closed by" links for action items.

**Fixed along the way:** proposals reported as decisions (better with the stronger model); a false "final
design had an LCD" claim (caught by code against memory); an answer with no citations; roles taken from
"Are we all gonna draw a cat?".

## Tools and where they broke
- **No framework:** I wanted every step visible and testable.
- **Models chosen by measuring:** gpt-4.1-mini for extraction (as good as gpt-5-mini, 2.5× cheaper); gpt-6-luna
  for the agent (+7 points at ¼ the price).
- **Where it broke:**
  - The extractor confused fact ids with turn numbers, and once got stuck repeating itself.
  - 45% of its update links pointed at the wrong subject.
  - gpt-6-luna only does tool calling with reasoning switched off (on Chat Completions), and it writes its own
    citation markers into answers. Code strips them.

## Production needs
- Postgres + pgvector instead of SQLite, with ingest through a queue.
- Real speaker names, access control, privacy and retention rules.
- A growing human-labelled test set with paired runs.
- Next: the Responses API (so gpt-6-luna can reason), a check that the answer agrees with its own citations,
  and a final verifier call.
