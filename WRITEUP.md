# Meeting memory agent: write-up

**What it does:** it reads 138 meeting transcripts once, stores them as searchable text, vectors and dated
facts, and answers questions with a tool-calling agent. Every quote in an answer is checked by code.
**Stack:** plain Python, the OpenAI SDK and SQLite. Memory extraction uses gpt-4.1-mini; the agent uses
gpt-6-luna; vectors use text-embedding-3-small.

## 1. Architecture

```
INGEST (once per meeting)                         ANSWER (per question)
transcript -> turns (regex)                       question -> agent loop (max 12 steps)
          -> memory.db: turns, ~200-word chunks      1. plan_searches (forced): model splits the question,
             + FTS5 keyword index                       code runs one search per part
          -> vectors.db: chunk embeddings            2. model calls get_memory / search / read_turns / get_roles
          -> facts.db: LLM extracts decisions,       3. submit_answer -> code checks quotes, roles, length,
             actions, numbers, disagreements,           empty citations, outcome claims vs memory; problems go
             roles, each with an evidence turn          back to the model; markers stripped from the text
```

- **Chunks are for finding things, turns are for proving them.** Half of all turns are under 6 words, so
  search runs over windows of turns. A citation is always one exact turn.
- **Hybrid search:** keyword (BM25) and meaning (embeddings) results are combined with Reciprocal Rank
  Fusion, filtered by team. On the development set, recall@5 was 40% for keyword only, 43% for vectors
  only and 47% for hybrid.
- **How it knows what's current:** each meeting is extracted in order (a→d), and the model sees that
  team's earlier facts. **Code** then decides what is current: the latest *settled* decision on a subject
  wins. An extractor link ("this updates K12") only counts if it is earlier, on the same subject, lists at
  most 3 ids, and isn't a proposal undoing a decision.
- **Brief:** the same loop with `submit_brief`. "Before the final meeting" is enforced in code: the tools
  can't see meeting d, and any citation from d is rejected. Code also gives the model a checklist of the
  settled decisions and open actions in memory.
- **At 10,000 meetings:** every tool is filtered by team, so each question costs the same at any corpus
  size. Ingest grows linearly, one pass per meeting.

## 2. Model vs code

| The LLM decides (needs judgment) | Code does (needs to be exact, cheap, repeatable) |
|---|---|
| What a messy exchange means: proposal or decision, who owns a task | Parsing, chunking, indexing; spoken numbers → digits; L_C_D_ → LCD |
| What to look up next, what the evidence says, how to phrase the answer | Running every tool, the team lock, the step limit, the brief's cutoff |
| Which points belong in the brief | Every quote (≥6 words, one turn, repaired if close); roles verified against their evidence turn |
| | What is current; outcome claims checked against memory; stripping citation markers from answers |
| | Writing the test answers: that was me, by reading |

**Rule I followed:** when a prompt rule only worked some of the time, I moved the check into code.
Examples: roles, the 1–3 sentence limit, empty citations, and quotes or markers inside the answer text.

## 3. Proof it works

**Test sets, all written by reading the transcripts, with 48 citations checked by code:**
- **Development:** 10 questions, 43 key facts. I used these while building, so the scores may be a little optimistic.
- **Held-out:** 6 questions, 29 key facts, on the 5 teams no other question set uses. Run after the code was frozen.

**Key facts found (my reading; the LLM judge's score in brackets):**

| | gpt-4.1-mini (4 runs) | **gpt-6-luna** |
|---|---|---|
| Development | ~67–74% (judge 60–79%) | **~81% (81%)**; 4–5/10 fully correct |
| Held-out | ~71–79% (48–69%) | **~76% (66%)** |
| All 72 facts | ~72% | **~79%** |
| Cost per question | ~$0.017 | **~$0.005** |
| Time per question | 16–18 s | 13 s |

- **Citations:** every answer has 1–3 valid quotes. Catalist's 25 have 70, all valid.
- **Memory tests** (deterministic, so they don't vary between runs):
  - `eval_current.py` checks whether memory knows the final state on 12 subjects: **6/12 before the memory fixes, 11/12 after**.
  - Roles verified against their evidence turn: 109/125.
- **Ingest cost (measured):** 138 meetings cost **$1.06**. That's **~$7.70 and ~80 minutes per 1,000 meetings** with 5 parallel workers, or about half with the Batch API.

**The judge is unreliable.** In one run it marked 5 facts missing that the answers state and credited one
that wasn't there. Every number above was checked by reading. A gain of a few points can't be measured on 72 facts.

## 4. Three questions it still gets wrong (gpt-6-luna)

1. **H04 (ES2012 chip) — it never checked what changed later.** It reports meeting c's custom chip and the
   costing, but not that meeting d switched to a standard chip. That change *is* in memory (eval_current
   confirms it), but the agent stopped after 5 tool calls without asking memory.
   **Fix:** code adds each subject's memory timeline to the plan step, so the change is in front of the
   model whether or not it asks.
2. **T02 (TS3012 kinetic) — part of the answer presented as the whole.** It names B's objection (people
   won't shake the remote), which is real, but misses the project manager's main one: remotes lie still
   99% of the time (meeting c).
   **Fix:** a checkable checklist of each part's evidence, plus a second check call.
3. **Brief — a resolved task listed as open.** It lists C's research on a rechargeable remote as an open
   action, although meeting c ruled the rechargeable unit out. Deciding open vs resolved across meetings
   is the hardest memory judgment.
   **Fix:** action items need a "closed by" link that code can check.

**Fixed along the way (each found in the logs):**
- **Proposal taken as a decision:** Q05, T07, and H06's invented budget cuts. Better with the stronger model.
- **The final remote "had an LCD" (T08):** caught by the code check against memory.
- **An answer accepted with no citations (T04):** a clearer error message and quote recovery in code.
- **Roles taken from "Are we all gonna draw a cat?":** code now verifies the role is actually stated.
- **Citation markers inside answers** (`[ES2008a]`, hidden Unicode characters): stripped by code.

## 5. Tools and where they broke

- **No framework:** a hand-written loop (agent.py, ~400 lines with all checks), so every step is visible, logged and testable.
- **Models were chosen by measuring:**
  - gpt-4.1-mini vs gpt-5-mini for extraction: similar quality, 2.5× cheaper and 4.6× faster.
  - gpt-4.1 skipped for cost.
  - gpt-6-luna for the agent: +7 points on 72 facts at ¼ the price.
- **Where things broke:**
  - Extraction: fact ids got confused with turn numbers (fixed with K-ids); IS1009b got stuck repeating itself (fixed with an output cap and one retry).
  - Memory: 45% of extractor links pointed at another subject, so code now checks every link.
  - gpt-6-luna on Chat Completions: tool calling only works with reasoning switched off (`reasoning_effort none`), and it writes its own citation markers into the answer.
  - The LLM judge was first too lenient, then too strict.

## 6. What production would need

- **Data:** one Postgres database with pgvector instead of three SQLite files; new meetings ingested through a queue (parse → embed → extract).
- **Identity and safety:** real speaker names from the calendar or meeting tool; access control on who can search which meetings; privacy and retention rules.
- **Quality:** a growing human-labelled test set with paired runs (counting facts that go wrong → right vs right → wrong), plus sampled checks of extracted facts and alerts.

## 7. Future improvements

- The memory timeline per subject in the plan step (fixes H04-type misses).
- A check that the answer agrees with its own citations.
- A second check call: is this settled or only proposed, and did it change later?
- Move the agent to the Responses API so gpt-6-luna can use its reasoning.
- Re-extract memory with a stronger model; add "closed by" links for action items.
- A shorter agent prompt, measured against the current one.
- New held-out questions. The current 6 shaped the memory fixes, so they're now half development set.
- The Batch API for ingest, shorter vectors at very large scale, and graph memory for questions across projects.
