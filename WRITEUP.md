# Meeting memory agent: write-up

**What it does:** it reads 138 meeting transcripts once, stores them as searchable text, vectors and dated facts, then answers questions about them with a tool-calling agent. Every claim in an answer is backed by an exact quote that code has checked.
**Stack:** plain Python, the OpenAI SDK and SQLite. No framework.

## 1. Architecture

```
INGEST (once per meeting)                        ANSWER (per question)
transcript -> parse turns (regex)                question -> agent loop (gpt-4.1-mini, max 12 steps)
          -> memory.db: turns + ~200-word chunks    tools: list_meetings, get_roles, get_memory,
             + FTS5 keyword index                          search (hybrid), read_turns, submit_answer
          -> vectors.db: chunk embeddings        submit_answer -> code checks every citation, role,
          -> facts.db: LLM extracts decisions,                    length and format; failures go
             actions, numbers, disagreements,                     back to the model to fix
             roles, each with an evidence turn
```

- **Chunks are for finding things, turns are for proving them.** Half of all turns are under 6 words ("Yeah."), so search runs over windows of turns. A citation is always one exact turn.
- **Hybrid search** combines keyword (BM25) and meaning (embeddings) results with Reciprocal Rank Fusion. On my 10-question set, recall@5 was 40% for keyword only, 43% for vector only and 47% for hybrid.
- **How it knows what's current:** each meeting is extracted in order (a→d), and the model sees that team's earlier facts with K-ids. It can mark a new fact as `updates: K12`. Then **code** decides what is current: the latest meeting where the fact is settled (agreed or rejected) wins, and any older fact it updates is marked superseded. The agent gets this as a timeline from `get_memory`.
- **Brief:** the brief uses the same loop with `submit_brief`. "Before the final meeting" is enforced in code: tools can't see meeting d, and citations from d are rejected. Code also builds a checklist of settled decisions and open actions from memory, then checks the brief covers them.
- **At 10,000 meetings:** nothing ever reads all the meetings at question time. Every tool is filtered by team, and each question costs the same whether there are 138 meetings or 10,000. Only ingest grows with the corpus, and it is linear: each meeting is processed once.

## 2. Model vs code

| The LLM does (needs judgment) | Code does (needs to be exact, cheap, repeatable) |
|---|---|
| Extract decisions, actions and disagreements from messy speech, telling a proposal from an agreement | Parsing, chunking, indexing, spoken numbers to digits ("fourteen point six" → 14.6), L_C_D_ → LCD |
| Choose what to look up next, read the evidence, write the answer | Run every tool, lock the team, the step limit, the brief's cutoff |
| Choose and phrase the brief's points | Verify every quote (≥6 words, one turn); repair near-misses; check roles against the roles table |
| | Decide what's current (meeting order + status + update links) |
| | Write the test answers. That was me, by reading, so the AI isn't graded by the AI |

**Rule I followed:** when a prompt fix only worked some of the time, I moved the check into code. Examples: wrong roles, answers longer than 3 sentences, turn numbers written in the answer text, answers with no citation.

## 3. Proof it works

**10 test questions:** I wrote these myself with 43 key facts and 30 citations that I verified by reading. The final code was scored by a strict LLM judge, then I checked the judge by reading.

| | Result (gpt-4.1-mini, final code) |
|---|---|
| Key facts | judge 30/43 = **70%**; my reading **~32/43 = 74%** (the judge was too strict on T01 and T09) |
| Fully correct answers | 2/10 |
| Citations | 29 cited quotes, all checked valid by code; every answer had at least one; 6/10 cite 2+ meetings |
| Cost / time | **$0.017** and **~18 s** per question, ~9 tool calls |
| Earlier runs | 79% / 78%. Run-to-run variation is about as big as a prompt change |

**Catalist's 25 questions:** all citations are valid and all answers are 1–3 sentences. When I read them against the transcripts, 2 are wrong (Q05, Q07) and 2 are partly right (Q18, Q25). I stopped re-running at that point, because tuning on the graded questions would just be fitting to the test.

**Ingest cost (measured):** 138 meetings cost **$1.06** ($0.03 embeddings + $1.03 extraction). That works out to **~$7.70 and ~80 minutes per 1,000 meetings** with 5 parallel workers, or about half the cost with the Batch API.

## 4. Three questions it got wrong

1. **Q05 (TS3007 energy source).** It took an early proposal as the first decision. It said the team "first chose a simple battery", but that was only proposed in meeting b. The first settled choice was a battery + kinetic **combination** (c, "the combination between battery and kinetic"), and kinetic was dropped in d. The same failure showed up on T07, where a prompt fix helped once. It isn't solved, because telling "maybe X" from "we go for X" is exactly where the model is weakest.
2. **T08 (IS1004).** It mixed an old option into the final design. The answer says the final remote had "an LCD screen with a scroll wheel", but the LCD was dropped in meeting c (d opens with "No L_C_D_"). It also missed the final simple beep. Memory had the right status, but the agent wrote from a search snippet about earlier options. **Fix:** code should check every "final" claim against `get_memory`'s current status.
3. **Q07 (IS1008 display cost).** It tied the number to the wrong thing. It reported "12–12.5 Euros" as the display's cost, but that is the team's total cost against the budget. The follow-up only says the display is "very expensive", so the honest answer is "no figure was given". **Fix:** extract numbers together with what they refer to, and let the agent say "not found".

**One it got wrong that code now catches:** on T05/T06 the agent guessed a speaker's role. `get_roles` now fills in a missing role by elimination, and code rejects any "Speaker X (Role)" that doesn't match the roles table.

**Brief limits:** all 40 citations are valid and none comes from meeting d. But "open disagreements" names the scroll-wheel, which was already settled; the real open issue was the missing cost information. And speech recognition's status is muddled.

## 5. Tools and where they broke

- **No LangChain or LangGraph:** the agent loop is about 200 lines, and I wanted every step visible and testable. I'd consider LangGraph once there are several agents or human approval steps.
- **gpt-4.1-mini** for extraction and the agent. I chose it on cost; I compared it with gpt-5-mini on extraction, and they caught different facts. **text-embedding-3-small** for vectors ($0.02/M tokens).
- **Where it broke:** the extractor confused fact ids with turn numbers (fixed with K-ids); it invented statuses (fixed with a mapping in code). It got stuck in a loop on IS1009b and returned cut-off JSON (fixed with a token limit + retry). The first judge was too lenient and the second sometimes too strict, so every judge score was checked by hand. A whole-team timeline got cut at 12k characters, so it now uses a compact view. Search silently fell back to keyword only when vectors were missing, so it now raises an error instead.

## 6. What production would need

- **Data:** one Postgres database with pgvector (HNSW index) instead of three SQLite files. Each new meeting is ingested through a queue (parse → embed → extract) instead of rebuilding everything.
- **Identity and safety:** real speaker names from the calendar or meeting tool; access control on who can search which meetings; privacy and retention rules.
- **Quality over time:** a growing labelled test set, run on every prompt or model change, plus sampled checks of extracted facts and alerts when error rates rise.

## 7. Future improvements

- A code check of "final / decided" claims against current memory status (fixes T08-type errors).
- A second extraction pass focused on proposals that were rejected or changed (fixes Q05 and T07).
- Numbers stored with what they refer to (fixes Q07).
- A stronger agent model on multi-part questions. Most remaining misses come from stopping too early (after ~6 tool calls).
- The Batch API for ingest (about half the cost), and shorter embedding vectors at very large scale, after measuring the recall cost.
- Graph memory (people ↔ decisions ↔ topics) for questions across projects.
