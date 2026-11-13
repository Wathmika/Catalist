# Build notes

A running log of what we built, what we decided and why. This becomes the
one-page write-up and the video script at the end.

## Assessment checklist (what Catalist is grading, and where we cover it)

| Criterion | Where we cover it | Status |
|---|---|---|
| Real agent behaviour (retrieve, loop, steps; holds at 10k meetings) | Phase 2 search, Phase 4 agent loop | done |
| Memory over time (what is *current*; references back) | Phase 3 extraction + time-ordered events | done |
| Tool-fit judgment (model vs code, defended) | Every phase: note it below as we go | done (WRITEUP s2) |
| Proof it works (10+ own questions, scored; cost + time for 1,000 meetings) | Phase 1b (first test questions), Phase 6 | done (WRITEUP s3) |
| Honesty about failure (3 wrong answers, explained) | Phase 6 | done (WRITEUP s4) |
| Platform reasoning (framework, models, APIs, where they broke) | Decided: plain Python + OpenAI SDK, SQLite. Log breakages as they happen | done (WRITEUP s5) |

### Scaling and production notes (collect here; goes into the write-up's "what would need to change")
- **Vector search compares in plain Python.** Fine for one team's ~120 chunks
  (milliseconds), too slow for millions of chunks without a filter. Production: a vector
  index (pgvector with HNSW, or Pinecone) that finds nearest neighbours without
  comparing every chunk.
- **The team filter is what keeps search small.** Here it's "team"; in a real company
  it would be project, participants or date range. Without some filter, cost and noise grow with the corpus.
- **SQLite is one file on one machine.** Production: Postgres (+ pgvector) so many users and
  workers can read and write at once.
- **store.py rebuilds everything** (1 second for 138 meetings). Production: ingest each new
  meeting incrementally as it arrives, through a queue.
- **Embeddings run one batch at a time.** Production: parallel calls, or OpenAI's Batch API
  (cheaper, slower). Vectors are cached by text hash, so nothing is ever paid for twice.
- **Vector size:** 1,536 numbers per chunk. At millions of chunks, request shorter vectors
  (the `dimensions` option, e.g. 512) to cut storage ~3x; check recall with eval_search.py first.
- **Three database files** (memory.db: code, rebuilt in 1 s; vectors.db: $0.03; facts.db: ~$1,
  non-deterministic) are separate so store.py's delete-and-rebuild never destroys paid work.
  Production: one Postgres database (+ pgvector) with these as tables, updated incrementally.
- **Speaker letters, not names.** Production: real identities from the meeting tool or calendar.
- **Not built, but needed for a real company:** access control (who may search which meetings),
  privacy and retention of personal data, monitoring answer quality over time.

### Future improvements (for the write-up / presentation section)
- Graph memory (people <-> decisions <-> topics over time) for cross-project questions.
- OpenAI Batch API for ingest (about half price; meetings don't need facts within seconds).
- Event-driven ingest: each finished meeting goes through a queue (parse -> embed -> extract).
- Real speaker identities from the meeting tool / calendar instead of letters.
- Monitoring: sample extracted facts, measure error rates, alert on drops.
- Re-run extraction when a better model arrives (raw transcripts stay the source of truth).
- Shorter embedding vectors at very large scale (after measuring the recall cost).
- A cheap per-team tidy-up pass after extraction (~$0.10 for all teams): merge same-subject
  topic names and re-link updates from the fact list alone. Only if the eval shows wrong answers
  caused by broken links / split topics (measure first).
- Smaller or fine-tuned model just for extraction, once there's labelled data.
- Try a stronger agent model (gpt-4.1) on the eval set: remaining errors are thoroughness on
  multi-part questions; not run here for cost reasons (~5x per question).
- Two complementary extractors merged (gpt-4.1-mini + gpt-5-mini caught different facts), or a
  second pass focused on proposals that were shot down (the weak spot found in Phase 3).
- Access control and privacy controls per meeting.

### Model vs code log (add a line every phase)
- Parsing transcripts into turns: **code** (fixed format, regex is exact, free, repeatable)
- Token counting, stats, missing-meeting check: **code**
- Speaker-letter consistency check, keyword finder: **code** (fast regex scans; a human judges the results)
- Building the database, chunking, keyword search, team filter: **code** (deterministic, free, ~1s for 138 meetings)
- Turning number words into digits and L_C_D_ into LCD for search: **code** (fixed rules)
- Extracting decisions / actions / numbers / disagreements / roles from messy speech: **LLM** (needs understanding: a proposal vs an agreement)
- Checking each extracted fact (turn exists, role said by that speaker, 'updates' points at a real fact), storing, ordering, deciding what's current: **code**
- Agent: choosing what to look up next, interpreting evidence, phrasing the answer: **LLM**
- Running every lookup, team / meeting guardrails, step limit, forcing a final answer, checking and repairing every citation: **code**
- Brief: which points matter and how to phrase them: **LLM**. The 'before the final meeting' cutoff, checking every point's citations and roles, dropping unsupported points, rendering: **code**
- Writing our own test questions and answers: **human** (us, by reading), never the LLM, or we'd be grading the AI with the AI

## Phase 1: Data exploration (`src/explore.py`, plain code, no AI)

**Numbers**
- 138 meetings, 35 teams, ~677k words, **817,755 tokens** exact with tiktoken (o200k_base,
  the tokenizer of current OpenAI models), avg ~5.9k tokens per meeting.
  The task sheet's "~880k" is close; older tokenizers split text into more pieces.
- 48,148 speaker turns. Median turn is **5 words**. **51% of turns are under 6 words**
- ~9% of all words are fillers (um, uh, yeah, okay)
- IS1002 has no meeting **a**; IS1005 has no meeting **d**

**Decisions that follow from the numbers**
1. **No "paste everything" approach.** 818k tokens doesn't fit in a prompt,
   and at 10,000 meetings it would be ~59M tokens. We retrieve instead.
2. **Search chunks = windows of several turns; citations = one turn.**
   Half of all turns are too short to search on their own ("Yeah.", "Okay."),
   but the graders require a quote of 6+ words from a single turn. So we search
   over groups of turns, but every chunk remembers its individual turns, so we
   can always cite one turn exactly.
3. **We never clean the text we store.** Fillers stay in, because quotes are
   checked word for word. (We may ignore fillers when *searching*, but the
   stored text stays original.)
4. **Missing meetings are real.** IS1002 has no meeting a (Q09 asks about its
   "first" price), so the agent must treat the team's earliest *available*
   meeting as the start and not assume a-b-c-d always exist.
5. **Cost baseline:** ~5.9M tokens to read 1,000 meetings once at ingest.

**Honest note for the interview:** one team's 4 meetings are only ~24k tokens,
which *would* fit in a prompt. We still retrieve, because in the real product
a team or company has hundreds of meetings, not four, and the brief explicitly
says the approach must hold at 10,000.

## Phase 1b: Understanding the questions and the speech (`src/inspect_data.py`, plain code)

**1. The 25 questions fall into 5 types (most questions mix 2-3):**

| Type | Example | Questions |
|---|---|---|
| Decision changed over time (first X, then Y, why) | Q02 case material switched | Q01-Q06, Q12, Q13, Q16-Q19, Q22-Q25 |
| Who said / proposed / objected (speaker + role) | Q03 who proposed replacing titanium | Q03, Q05-Q09, Q18-Q22 |
| Commitment and follow-up (someone said they'd check; what came back) | Q07 cost of a display | Q07, Q11, Q14, Q20, Q21 |
| Numbers (prices, costs, totals, budget) | Q15 total vs limit | Q07, Q09, Q14, Q15, Q17, Q21 |
| Why / reason | Q10 why that age group | Q01, Q02, Q06, Q08, Q10, Q12, Q25 |

**What this tells us to store (Phase 3):** decisions with their status over time
(proposed / chosen / dropped / replaced), action items (owner + what + whether it
came back), numbers (value + what it refers to), and a speaker -> role table per team.

**2. Are speaker letters the same person across a team's meetings?**
Checked two ways with regex: (a) self-introductions like "I'm the Industrial Designer",
(b) who uses chair words like "agenda" / "minutes" / "next meeting" most in each meeting.
- Every apparent conflict in (a) was a false alarm on reading, e.g.
  "I'm not a very good industrial designer" (ES2013b).
- In (b) the chair has the same letter across meetings in 27 of 35 teams; the other 8
  have weak signals (0-2 mentions), not real contradictions.
- **Conclusion:** letters look consistent within a team, **but the project manager
  is not always Speaker A** (ES2002: B, ES2005/ES2006: C, IS1000: B).
  So we build a speaker -> role table **per team**, never assume A = PM.
- Honest caveat: this is evidence, not proof. The agent should cite the
  self-introduction turn when it states someone's role.

**3. How people actually talk (feeds the Phase 3 extraction prompt)**
- **Numbers are always spoken as words.** 0 euro amounts written as digits,
  417 written as words ("twenty five Euros", "twelve and a half"). Keyword search
  for "25" or "12.5" finds nothing. The extractor must turn words into numbers,
  while quotes keep the words exactly.
- Acronyms are spelled with underscores: "L_C_D_", "T_V_", "P_D_A_". Search for
  "LCD" misses them unless we normalise for search (stored text stays original).
- Decisions are rarely clean: "Okay, so we're definitely going for rubber",
  "No r no scroll-wheel. Okay." They often come after a long back-and-forth.
- Back-references without names are common (~280 turns): "what we discussed last time",
  "like we agreed before". Answering needs earlier meetings, which is why
  memory is ordered by meeting.

**4. Our own test set** (`tests/my_questions.json`, checked by `tests/check_quotes.py`):
10 questions, answers written by reading the transcripts, 30 citations all verified
word for word by code (6+ words, inside one speaker turn).

| ID | Team | Type |
|---|---|---|
| T01-T04 | TS3012 | decision changes, who objected, why (the brief team) |
| T05 | ES2005 | numbers: over budget, how they fixed it |
| T06 | TS3008 | disagreement: who argued what, final target group |
| T07 | ES2015 | who proposed + numbers (component cost, total vs budget) |
| T08 | IS1004 | decision against a feature, reasons, what replaced it |
| T09 | IS1004 | open question carried to the next meeting and its outcome |
| T10 | TS3005 | why one option won, raised again later and rejected |

Rules we followed:
- **None of these teams are in Catalist's 25 questions**, so the test is independent
  and we can't accidentally tune the agent to their questions (overfitting).
- **Answers come from a human reading the source, never the LLM**, otherwise we'd
  be grading the AI with the AI.
- Each question has `key_facts`, a checklist we score the agent against in Phase 6.

What TS3012 tells us for the brief: LCD dropped in b, back in c, kept in d;
kinetic energy rejected for regular batteries; scroll-wheel cut; rubber kept
over the yellow colour when over budget; speech recognition dropped in d for cost.

## Phase 2: Memory store and keyword search (`src/parse.py`, `src/store.py`, `src/search.py`, plain code)

**What we built**
- `data/memory.db` (SQLite, ~26 MB, rebuilt from the transcripts in ~1 second):
  - `meetings`: team, letter, **ord** (a=1..d=4). The order is how the agent knows what's most recent.
  - `turns`: 48,148 rows, **original text, never modified**. Source of truth for quotes.
  - `chunks`: 4,196 groups of consecutive turns (~200 words, 2-turn overlap). Each remembers
    `start_turn..end_turn`, so a search hit always leads back to exact single turns.
  - `chunks_fts`: keyword index (SQLite FTS5, BM25 ranking, porter stemming so
    "batteries" matches "battery").
- `search.py`: the only door the agent uses: `search(query, team, meetings, k)`,
  `get_turns(meeting, start, end)`, `list_meetings(team)`.

**Design decisions**
- **Two copies of the text.** Stored text is original (for quotes); a separate *search*
  copy turns `L_C_D_` into `LCD` and spoken numbers into digits
  ("fourteen point six" -> 14.6, "twelve and a half" -> 12.5), from the Phase 1b findings.
- **Chunks never split a turn**, so every hit maps cleanly to whole turns for citing.
- **Overlap of 2 turns** so a question and its answer split across a chunk boundary are still found together.
- **Filter by team before ranking.** At 10,000 meetings the search only ranks one team's
  ~120 chunks, not millions. In production the filter would be project, people or dates.
- **One search interface.** The agent never touches SQL, so the store can be swapped
  (pgvector, Pinecone) by changing `search.py` only.

**First measurement** (`tests/eval_search.py`): send each test question *as written* to
keyword search with the team filter, and check whether the top-k chunks contain the turns
our citations come from.

| | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| Keyword search, raw question | 40% | 53% | 67% |

Why it misses (read from the failures):
- **The question's words aren't the answer's words.** T04 asks about "case material";
  the evidence says "rubber". Keyword search can't connect them.
- **Generic words dilute the query** ("wanted", "changed", "came").
- We did **not** tune stopwords against these 10 questions: that would be overfitting
  to our own test set.

Next: meaning-based search (embeddings) to catch "material" -> "rubber", combined with
keyword search. And in Phase 4 the agent writes its own queries and searches several times.

## Phase 2b: Embeddings and hybrid search (`src/embed.py`, `src/search.py`)

- `embed.py` turns each chunk into an OpenAI embedding (`text-embedding-3-small`, 1,536
  numbers), stored in a separate `data/vectors.db` keyed by a SHA-1 hash of model + text:
  rebuilding memory.db never loses paid work, and identical text is never embedded twice.
  Batches of 100, saved after each batch, resumable. Logs real tokens, cost and time.
- `search.py` modes: `keyword` (BM25), `vector` (dot product = cosine, team filter first),
  `hybrid` (default: top 30 of each merged with Reciprocal Rank Fusion, 1/(60+rank),
  because BM25 and cosine scores are on different scales).
- Falls back to keyword with a printed reason if embeddings are unavailable. A team with
  no vectors at all raises an error instead of silently becoming keyword-only
  (found on review: it would have made hybrid look no better than keyword).
- **Model vs code:** the embedding model only turns text into numbers; ranking, filtering,
  merging and fallback are code.

| | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| keyword | 40% | 53% | 67% |
| vector | 43% | 60% | 77% |
| hybrid | **47%** | 60% | 73% |

Reading it honestly:
- Meaning search helps: recall@10 53% -> 60%, recall@20 67% -> 77%.
- **Hybrid is best at top 5**, which is roughly what the agent reads per search, so
  **hybrid is the default**. Keyword also covers exact names and numbers ("titanium",
  "14.6"), which our 10 questions under-represent.
- **The test set is small:** 30 evidence turns, so one turn = 3.3 points. Differences of
  a few points are within noise; the trend (meaning search helps) is the finding, not the decimals.
- This is the question-as-written baseline. The agent writes its own queries and
  searches several times, so its real recall should be higher.

**Why we need an agent, with evidence (T04).** T04 asks three things at once (material, why,
what they sacrificed when over budget); evidence is in c #107, c #286 and d #136.
- One search with the full question: 0/3 in the top 5 (keyword), 0/3 (hybrid).
- Two focused searches, as an agent would write them: "rubber plastic wood case material"
  finds c #107 and c #286 in the top 5; "keep rubber or yellow colour over budget" finds d #136 ranked 1st.
- So: single-shot retrieval breaks on multi-part questions; an agent that splits the question,
  searches per part and uses what it learns (they chose rubber) for the next search finds all of it.
  Candidate for the "questions it got wrong" section if the agent still misses it.

Embedding cost/time for 138 meetings (real run): **1,446,847 tokens, $0.029, 39 seconds**
(one process, batches of 100). Per 1,000 meetings: ~10.5M tokens, ~$0.21, ~5 min.
More tokens than the raw transcripts (~818k) because chunks overlap by 2 turns and
carry "#12 A:" labels; that overhead is the price of never losing an exchange at a chunk edge.

Dimensions: kept the default 1,536. text-embedding-3 models can return shorter vectors
(e.g. 512) with a small quality loss; at our size (~25 MB of vectors) there's nothing to
gain, so we didn't trade accuracy for it. At millions of chunks, shorter vectors would cut
storage and search time; measure the recall drop with eval_search.py before switching.

## Phase 3: Memory extraction (`src/extract.py` LLM + code, `src/memory.py` code)

**Requirement it serves:** "Memory over time: a decision in meeting c can replace one from
meeting a ... Show how your agent knows what is current." Also the TS3012 brief, whose four
sections (decided / changed / open action items + owners / unresolved disagreements) map
directly onto the fact types. Extraction itself is our design choice for meeting these.

**How it works**
1. `extract.py`: the LLM reads ONE meeting (turn-numbered) plus the team's facts so far, and
   returns JSON (OpenAI structured outputs, strict schema): facts of type decision /
   action_item / number / disagreement, each with topic, status, who, value+unit, ONE evidence
   turn, and `updates` = the earlier fact it changes. Plus roles from introductions.
2. Code validates: turn exists; a role must come from that speaker's own turn; `updates` must
   point at a real earlier fact (else the link is removed). Issues are logged per meeting.
3. Meetings of a team run strictly in order (a->d) so each sees earlier memory; teams run in
   parallel (5 workers). Facts go to `data/facts.db` (separate from memory.db). Resumable;
   records model, prompt version, tokens, seconds per meeting.
4. `memory.py` (code only): groups facts by topic, sorts by meeting order and turn, and marks a
   fact superseded if a later fact `updates` it or a later decision on the same topic is
   settled (agreed / rejected). `up_to_ord=3` gives "what was known before meeting d" for the brief.

**Why one meeting per call isn't "pasting everything":** a fixed ~6-8k tokens per meeting, once,
at ingest. Pasting the corpus per question grows without limit.

**Cost estimate (before running):** 10,000 meetings x ~8k in / ~1.5k out: gpt-4.1-mini ~$56,
gpt-5-mini ~$50, gpt-5-mini via Batch API ~$25. About half a cent per meeting, paid once.
Our 138 meetings: under $1.

**Prompt v2 (`src/prompts.py`), upgraded before the first real run:**
- Sections in XML-style tags (role, context, task, definitions, rules, example, output).
- Every status defined; "one person liking an idea is proposed, not agreed".
- One worked example from an INVENTED meeting (teaches format without leaking test answers).
- Transcript and earlier memory sent as tagged data, declared not to be instructions.
- Schema field descriptions; temperature 0 for non-reasoning models.
- **`evidence`: 6-25 exact words from the turn. Code verifies: wrong turn -> corrected to the
  nearest turn that contains the words; words nowhere -> fact dropped.** Also code checks the
  status is valid for the type (e.g. a number can't be "agreed").
- Later, these verified evidence strings can be used directly as citations.

**Model choice: measured, not assumed.** Run TS3012 with gpt-4.1-mini (non-reasoning, $0.40/$1.60
per M tokens) and gpt-5-mini (reasoning, $0.25/$2.00) and compare against the known story.

**Run 1 (gpt-4.1-mini, prompt v2, TS3012):** 97 facts, 4 roles, $0.033, 121 s.
Real cost: ~12k input tokens per TS3012 meeting (long meetings + prompt + growing memory),
so **~$8 per 1,000 meetings** standard, ~$4 with the Batch API. Our 138 meetings: ~$1.

31 issues in b-d. Reading them showed **our evidence check was too strict, not the model too
sloppy**: the model copied "almost exactly" (dropped an "uh", wrote "the the the" for
"the the the the", "Okay, okay," for "Okay. Okay,", or put "..." in the middle). That threw
away good facts, including the key decision "Okay. Okay, so we're definitely going for rubber."
Also: `updates` sometimes listed several ids ("F26,F27"); a few decisions got status "open".

**Fix (prompt v3 + code):**
- Code **repairs** near-miss evidence: finds the span of the real turn that best matches
  (word-level similarity, ignoring case/punctuation, >= 0.75), searching the claimed turn and
  its neighbours first, then the meeting, and stores the ORIGINAL words. Tested on the real
  failures: rubber decision 1.00, "under twelve and a half" 0.95, "squishy feeling" 0.91;
  a made-up sentence and quotes stitched from several turns are still rejected.
- `updates` accepts several ids; unknown ids removed. Obvious status slips mapped
  (decision open -> proposed), others still dropped.
- Prompt v3: keep fillers, repeats and punctuation, no "...", one turn only; avoid 1-word turns.
- Raw model JSON is stored per meeting, so the checks can be re-run without paying again.

**Run 2 (prompt v3, TS3012):** 128 facts, $0.036. Only 2 facts dropped (both 1-word or
stitched quotes); 6 near-miss quotes repaired, 6 turn numbers corrected.

Checked against the TS3012 story we know by reading:
- Rubber agreed in c (#286), kept over the yellow colour in d (#136): correct.
- Scroll-wheel rejected in c (#635): correct. Wood rejected (#101): correct.
- Regular batteries agreed in c (#75): correct, but the kinetic proposal and the PM's
  objection (c #47, #56) were NOT extracted. The agent must still use search for "who objected".
- Speech recognition: proposed b, rejected b, discussed c, scrapped d for cost: correct story.
- LCD: rejected b (#112), back in c (#168, stored as "proposed"), agreed with speech rec. (d #2).
- Budget 12.5 EUR (d #48), B's disagreement about the colour (d #47): correct.
- Roles A=PM, B=UI, C=ID, D=Marketing, each with its introduction turn: correct.

Two problems found:
1. **Hallucinated update links.** Many `updates` pointed at ids that didn't exist (F416, F618):
   the model mixed up fact ids with turn numbers (both shown as numbers). Code removed them,
   but links were lost.
2. **Topic fragmentation.** The same subject under several names ("lcd screen", "lcd screen
   size", "lcd screen and speech recognition"), so "current" was marked per name, not per subject.

**Fix (prompt v4 + code):** known facts get short local ids K1, K2 ... (mapped back to database
ids by code; anything else is removed); the prompt lists "Existing topics (reuse these names)"
and asks for one topic per subject. `memory.py` can show a subject's facts as one
chronological list (substring match, e.g. "lcd"), so the latest fact is simply last.
Honest limit: "current" is a code heuristic over the model's labels; the agent still reads
the dated timeline and the cited turns, not just the CURRENT flag.

**Run 3 (prompt v4, gpt-4.1-mini, TS3012):** 98 facts, 42 topics (cleaner: one "lcd screen
inclusion", one "remote control colour", one "speech recognition"), $0.036.
The LCD story now comes out as one linked, time-ordered list ending in the d decision.

New check script `tests/eval_memory.py`: 14 moments of the TS3012 story we verified by reading
(meeting + turn +/-3 + a key word) plus the 4 roles. Score: **11/14 story, 4/4 roles.**
Misses: PM dropping the LCD in b (#112; it WAS found in run 2, so run-to-run variation),
and the kinetic proposal + PM objection in c (#47, #56; missed in every run so far).
Lesson: extraction is good at decisions that get settled, weaker at proposals that get
shot down. So the agent uses memory for the timeline and search for the details.

**Model comparison (same prompt v4, same TS3012, same 14 checks):**

| | gpt-4.1-mini | gpt-5-mini |
|---|---|---|
| Story checks | **11/14** | 10/14 |
| Roles | 4/4 | 0/4 (returned none; the 4/4 first shown was stale data from the earlier run, found and fixed) |
| Facts | 98 | 76 |
| Tokens out | 7.6k | 38.3k (mostly hidden reasoning tokens) |
| Cost (4 meetings) | **$0.036** | $0.089 (2.5x) |
| Time | **127 s** | 582 s (4.6x) |
| Per 1,000 meetings | **~$9** | ~$22 |

They miss DIFFERENT things: gpt-5-mini caught the kinetic proposal and objection that
gpt-4.1-mini never does, but missed the scroll-wheel, speech recognition in b and d, and roles.
With one run each and 14 checks, 11 vs 10 is within noise; cost and speed are not.
**Choice: gpt-4.1-mini** (as good on our checks, 2.5x cheaper, 4.6x faster, reliable roles).
Future improvement: run two cheap extractors and merge (they're complementary), or a second
"what proposals were rejected?" pass.

Bug found by the comparison: re-extracting a meeting kept roles from the previous run
(INSERT OR IGNORE), so gpt-5-mini looked 4/4 on roles. Fixed: re-extracting a meeting now
clears that meeting's roles first.

**Full run (gpt-4.1-mini, prompt v4, all teams):** 135 of 138 meetings in **627 s** (5 teams in
parallel), **1,365,280 in + 285,694 out tokens, $1.00** -> **~$7.40 per 1,000 meetings**.
3,516 facts. TS3012 check after the full run: 11/14 story, 4/4 roles (same as before: stable).

Issues across all meetings (code checks at work):
- kept after repair: 214 evidence repaired, 97 turn numbers corrected, 47 statuses fixed
- dropped: 95 evidence too short, 61 evidence not found anywhere, ~42 invalid status,
  13 roles not said by that speaker. About 6% of facts dropped, i.e. caught, not stored wrong.
- 236 `updates` links removed because they pointed at K-ids that weren't shown (e.g. K133 when
  only ~60 facts existed). Links are a weak spot; "current" therefore also uses meeting order
  and settled status, not links alone.
- Roles: all 35 teams have roles; 12 teams have only 2-3 of 4 (not every speaker states theirs).

**Failure: IS1009b** returned broken JSON (cut off at ~145k characters): the model got stuck
repeating itself until it hit the output limit. Because a team's meetings depend on the
previous ones, IS1009 c and d were skipped (by design). Fix: cap output at 12k tokens; if the
output is cut off or not valid JSON, retry once with temperature 0.3 and a frequency penalty
(breaks repetition loops); failed attempts' tokens are still counted in the cost.

**IS1009 re-run after the fix:** b, c, d extracted (44 / 22 / 43 facts) in 80 s. No retry was
needed this time (the earlier loop didn't recur, so it was intermittent; the guard stays as
insurance). **All 138 meetings done: 1,397,367 in + 294,542 out tokens, $1.03, ~$7.50 per
1,000 meetings, ~11 minutes total with 5 teams in parallel.**

**Total ingest cost for 138 meetings:** embeddings $0.03 + extraction $1.03 = **~$1.06**.
Per 1,000 meetings: ~$0.21 + ~$7.47 = **~$7.70**, about 6 minutes of embedding (one process)
and ~80 minutes of extraction at 5 parallel workers (more workers or the Batch API: faster / cheaper).

**Open decision:** commit data/facts.db to GitHub for reproducibility (LLM output is never identical twice; 2.3 MB) or not. Decide before submission.

## Phase 4: The agent (`src/tools.py`, `src/agent.py`, `src/verify.py`, `src/run_questions.py`)

**Design**
- Hand-written tool-calling loop on the OpenAI Chat Completions API (no framework): the LLM picks
  tool calls, code runs them and returns small results, repeat; max 12 rounds, then code forces
  `submit_answer`, so the loop always ends.
- Tools (all plain code, results capped at ~12k chars): `list_meetings`, `get_roles`,
  `get_memory` (dated timeline with verified evidence quotes, topic filter, cutoff meeting),
  `search` (hybrid), `read_turns` (exact words, max 30 turns), `submit_answer`.
- **Guardrails in code, not in the prompt:** tools refuse other teams; in brief mode they refuse
  meetings after the cutoff (search is silently limited to allowed meetings).
- **Citations:** code checks every quote with the grader's rules (`verify.py`). Close-but-not-exact
  quotes are repaired to the real words (similarity >= 0.85, whole meeting); failures go back to the
  LLM with the reason (2 fix rounds); anything still failing is removed.
- Prompt (`prompts.py`, AGENT a1): method = split the question, timeline first, focused searches per
  part, read around hits, decide what is current, answer every part; strict citation rules.
- Every question logs its full trace to `logs/<id>.json` (tool, args, result size, preview).
- `tests/eval_agent.py`: runs the agent on our 10 questions; key facts judged by an LLM against the
  human answer key (it only compares, never decides truth; every judgement is printed), citations
  and evidence-near-our-key checked by code. Saves `tests/results_<model>.json`.

Tested here with a scripted fake model: tool routing, team refusal, repair of a near-miss quote,
rejection + fix of an invented quote, forced final answer.

**Run 1 (gpt-4.1-mini, agent prompt a1, our 10 test questions): $0.185 total, avg 9.6 tool calls,
15 s per question. 29 valid citations, every answer has at least one; 7/10 cite 2+ meetings.**
T04 (one search found 0/3 evidence turns): agent did memory -> focused search "costs rubber" ->
read_turns d #130-148 -> 3/3 valid citations across 2 meetings, 5 tool calls, $0.005.

LLM judge said 86% of key facts (37/43), 7/10 fully correct. **Reading the answers myself, the
judge was too lenient:** it counted "young users like the spongy feel" in T04 although the answer
never says it, accepted a wrong role in T06 and "no room on the banana shape" in T09. My manual
score: ~34/43 = **~79%**, 3-4/10 fully correct. Lesson: an LLM judge needs strict, explicit rules
and a human check. Judge prompt tightened; `--rejudge` re-scores saved answers without re-running the agent.

What went wrong (read from logs/T0x.json):
- **T06 wrong role:** TS3008's D never states a role (roles table had A, B, C). The agent guessed
  "D = User Interface Designer" (wrong; C is). Fix (code): get_roles fills the one missing role by
  elimination and labels it "deduced"; if 2+ are missing it says "give only the letter".
  (Also confirms Phase 1b: ES2005's C is the PM.) Prompt: roles only from get_roles, never guessed.
- **T07 proposal taken as decision:** read "Maybe that could be the backup ... solar backup" (c) and
  answered "they kept a solar backup"; the team actually went fully kinetic. Prompt: a proposal is
  not a decision, check what happened to it later before saying it was kept.
- **T05 wasted fix rounds + invented reason:** tried to cite "Eleven uh eleven Euros sixty" (5 words)
  three times; missed the duplicated-cost explanation and added a reason ("speech recognition
  breakthrough") not supported by what it read. Prompt: if the key turn is under 6 words, cite a
  nearby longer turn; every reason must come from a turn actually read.
- **T08 incomplete:** never searched the final design's "beep" (searched "final remote control" once).

Agent prompt a2 = these fixes.

**Run 2 (a2, all 10 as a regression check): $0.157.** Strict judge 34/43 = 79%, 5/10 fully correct;
my own reading ~33.5/43 = 78% (strict judge now agrees with a human within ~1 point).
- Fixed: **T06** role now right (D = Marketing, deduced); **T07** now "decided against a solar backup,
  fully kinetic, 3 EUR, 12.20 total" (5/5).
- Still wrong / new: **T05** called ES2005's C "Industrial Designer" although get_roles says C = PM
  (deduced); **T08** still misses the final "simple beep"; **T09/T10** dropped parts they had in run 1
  (T10 used only 5 tool calls and said wood "was not mentioned as a serious option": it was, and was
  rejected again in d).
- **Lesson: run-to-run variation is as big as the prompt effect.** Same overall score, different
  questions right. With 10 questions x 1 run, small prompt changes can't be told apart; the reliable
  improvements are the ones enforced by code.

**So a3 moves two checks from prompt into code:**
- `submit_answer` must include a **checklist of the question's parts**, each with the supporting
  meeting + turn. Parts marked "not found" are sent back once with "search once more".
- **Code checks every "Speaker X (Role)" in the answer against get_roles** (stated or deduced); a
  wrong or unknown role is rejected with the correct one. (Catches T05-style errors.)
**Run 3 (a3, only the 4 weak questions, ~$0.09):** my own reading
- T05: role now right (C = Project Manager): **the code role check works**. Still misses why the cost
  dropped (duplicated double-curve entry).
- T08: ~3/4 (strict judge said 1/4, too strict: "lack of confidence" and "background noise" are there).
  Still misses the final "simple beep".
- T09: ~2/5, says LCD dropped "by meeting d" (it was c); stopped after 6 tool calls.
- T10: ~3/5, 6 tool calls, misses wood rejected again in d.
- The parts checklist barely helped: the model fills it in with weak evidence instead of searching more.

**Where the agent stands (gpt-4.1-mini): ~75-80% of key facts, 5/10 fully correct, every answer with
valid citations, ~$0.016 and ~14 s per question.** Remaining errors are about thoroughness (stopping after
5-6 tool calls on 3-4 part questions), not invented quotes or wrong roles (those are now caught by code).
Judge note: the strict judge is sometimes too strict (T08), the first one too lenient; every number in
this section was checked by reading the answers.

**Decision: stay on gpt-4.1-mini for the agent** (cost). A gpt-4.1 run on the 10 questions (~$1) was
considered and not done; listed under future improvements with the expectation that a stronger model
mainly helps thoroughness on multi-part questions.

**Catalist's 25 questions, run 1 (a3):** code review of answers.json (citations re-verified, meetings
counted, team, length, roles): all citations valid; **Q16 had no citations and turn numbers in the
text**; Q25 had 4 sentences; 9 answers cited a single meeting. Reading the answers against the
questions: **Q05 contradicts itself** (first = battery, end = battery, yet "changed"); **Q25 contradicts
its own citation** ("broccoli preferred" vs quote "the broccoli would be out"); Q03 muddled (right
proposer B, right outcome "titanium gone" per TS3009d #408/#419, but it cites "Titanium stays there");
Q07's figure (12.50) looks like the budget, not the display cost; Q18's third part is weak.
Fix (code, in submit_answer): reject > 3 sentences, turn numbers in the text, zero valid citations
(one extra round); nudge once when all citations come from one meeting. Re-run only Q03 Q05 Q07 Q16 Q18 Q25.

**Re-run of Q03 Q05 Q07 Q16 Q18 Q25 (new checks):** all now have valid citations and 1-3 sentences.
Checked by reading the transcripts:
- Q03 fixed: B proposed titanium-coloured plastic (TS3009d #363/#366); titanium ditched (#408, #419).
- Q16 fixed: joystick favoured, push-buttons in the end, special colours given up (1 citation only).
- Q25 better: "the broccoli wins" (ES2011d); the reason is comfort (#222 "you can hold it"), the
  answer gives the fashion trend instead: partly right.
- **Q05 still wrong (verified):** TS3007 first settled on a battery + kinetic COMBINATION (c #337
  "the energy, the combination between battery and kinetic"), ended with batteries only after the PM
  (A) proposed skipping kinetic (d #369). The agent says "first: a simple battery" (a proposal in b).
  Same failure mode as T07: an early proposal taken as the first decision. Kept as a documented failure.
- **Q07 likely wrong:** "twelve to twelve and a half" is the team's total cost vs the budget, not the
  display's cost (IS1008c #143/#154: Ed asked to find out; d #74: "L_C display is gonna be very
  expensive"). Kept as a documented failure.
- Q18 still weak on the third part (who suggested how to present it to buyers).
Decision: stop re-running. Tuning answers on Catalist's own questions would be fitting to the test;
these become honest, verified failures for the write-up.

## Phase 5: The TS3012 brief (`src/brief.py`, prompt BRIEF b1)

- Same tool loop as the agent, with a final tool `submit_brief` whose points are grouped into the four
  sections Catalist asked for (decided / changed / open actions + owners / open disagreements), each
  point one sentence with 1-2 citations.
- **"Before their final meeting" is enforced in code:** cutoff = the team's second-to-last meeting
  (TS3012c). Tools hide or refuse meeting d (get_memory up to c, search limited to a-c, read_turns
  refuses d), and any citation from d is rejected. Tested: a TS3012d citation was rejected, as was a
  wrong role; the fixed point was accepted.
- Code checks every point's citations and roles; failing points go back to the LLM (2 rounds), points
  still without a valid citation are dropped. Output: answers.json "brief" + brief_TS3012.md + logs/brief_TS3012.json.
- Found while building: the whole-team timeline was cut off at 12k characters (TS3012 a-c alone is
  ~13k). get_memory without a topic now returns a compact timeline (no evidence quotes, up to 20k);
  quotes come from get_memory with a topic or read_turns.

**Brief run 1 (b1, gpt-4.1-mini):** a-c only (cutoff worked), every point cited, 4 sections filled.
Checked against what we know of TS3012 a-c, it **left out three settled decisions the memory had**:
energy = regular batteries (c #75), scroll-wheel rejected (c #635), colour = yellow (c #593). It also
listed "C researches rechargeable remotes" as open although c #65 ruled the rechargeable unit out,
cited a role introduction as an "action", and filled "decided" with generic project facts.

**Fix (b2): code-built checklist + coverage check.** brief.py reads the memory (up to c) and gives the
model (1) the settled decisions that must be covered and (2) the action items still marked open, with
the instruction to check whether each was later done or ruled out. After submit_brief, code checks each
settled topic appears in the brief (keyword match) and sends the gaps back once. Tested on run 1's
text: flags energy source, scroll-wheel, product colour (plus a few minor topics). Prompt: concrete
product choices first, no generic project facts, cite the turn where a task is given.
Honest limit: the keyword check is approximate (e.g. "curving" vs "curves"), so it nudges, it doesn't judge.

**Brief run 2 (b2):** 40 citations, all re-verified by code, none from meeting d. Now covers the
settled decisions run 1 missed: regular batteries (c), yellow cover with black buttons (c), no
scroll-wheel (c); plus rubber, LCD in the deluxe edition, square buttons, triangle menu buttons, beep
locator. The ruled-out rechargeable task is no longer listed as open. Open actions cite the turns where
the tasks were given (B + C final design, D market analysis).
Still weak (documented, not re-run):
- "Disagreements still open" names the scroll-wheel, which was settled; the real open issue going into
  meeting d is the missing cost information ("too bad we don't have the finance", c), which run 1 had.
- Speech recognition sits in "decided" with a muddled status (rejected in b, back with the deluxe
  edition in c).
- 14 "decided" points, some minor (power button on top); some citations are weak but valid.
Decision: keep b2. It's the agent's output, all claims are cited, and these limits go in the write-up.

## Phase 6: Final evaluation (final code: a3 + format checks + compact memory view)

**eval_agent.py, gpt-4.1-mini, all 10:** judge 30/43 = 70%, 2/10 fully correct; 29 valid citations,
0 answers without a citation, 6/10 cite 2+ meetings; $0.166 total ($0.017/question), 9.2 tool calls, 18 s avg.
Checked by reading:
- Judge too strict on T01 (answer does say the PM first wanted to drop the LCD) and T09 (answer does say
  dropped in meeting c). My reading ~32/43 = 74%.
- **T08 real error (verified):** says the final IS1004 remote had "an LCD screen with a scroll wheel";
  the LCD was dropped in c (IS1004d opens "No L_C_D_"). Still misses the final simple beep. Stale/mixed state.
- Variation: earlier runs 79% / 78%; this run 70-74%. Run-to-run noise is as big as prompt changes.
WRITEUP.md written with these numbers; 3 wrong = Q05, T08, Q07 (+ T05/T06 roles, now caught by code).

## Phase 6b: Two code fixes from the final run (agent prompt a4)

- **Sync check (connection back):** laptop had lost two fixes (VS Code revert): extract.py's IS1009b
  retry guard and search.py's "no embeddings -> error". Restored into the laptop versions (user's comments kept).
- **Fix 1: plan_searches, forced first step.** The model splits the question into parts with one query each;
  code runs every search (3 passages per part, 1,200 chars) via run_tool, so guardrails still apply.
  Why: logs show the model stopping after ~6-9 calls on 4-5 part questions. Cost: ~2-5k extra tokens carried per round.
- **Fix 2: outcome claims vs memory (check_outcomes).** Sentences with outcome words (final, decided, kept...)
  but not history words (first, proposed, considered...) are matched to the team's CURRENT settled
  decisions by topic keywords (l_c_d_ -> lcd; generic words and mixed "X and Y" topics skipped). Polarity
  (negation within 5 words before, same clause) vs status agreed/rejected; a mismatch is shown to the
  model once with the memory fact; the model decides. Tested on all 35 existing answers (25 Catalist +
  10 test): fires only on T08 (the real error); first version had 6 false alarms on the 25, fixed.
  Honest limit: keyword heuristic, tuned on these same answers.
- Dry run with a scripted fake model: plan forced in round 1, contradiction caught, corrected answer accepted.
- Next: run eval twice on gpt-4.1-mini + once on gpt-4.1 with this code.

## Held-out set (tests/heldout_questions.json, written after the code was frozen at a4)

6 questions on the only 5 teams no question set uses (ES2012, ES2013, IS1000, IS1005, IS1006), same types
as before: energy sources + why solar ruled out (H01), titanium -> plastic+rubber + budget (H02), promise to
check speech-recognition cost and the follow-up (H03), custom -> standard chip + cost (H04), radio vs
infra-red who/why/outcome (H05), energy change + how they "got under budget" (H06: by typing negative
numbers into the cost sheet). 18 citations, all pass check_quotes.py --heldout.
Found while writing: memory's role for ES2013 B is wrong (says UI Designer from a turn about drawing an
animal; B is Kate, the Industrial Designer). The code role check will push the agent to the wrong role:
an expected failure caused by memory, not the agent. Run once: python tests/eval_agent.py --heldout.

**Held-out run (gpt-4.1-mini, a4, run once):** judge 14/29 = 48%, 0/6 fully correct; 15 valid citations, 0 answers
without one; $0.136, 9.2 calls, 24 s. My reading ~20.5/29 = ~71% (judge very strict: H02 does give the iPod reason
and "too expensive / hard to shape"; H03 does give 3 Euros; H05 gives the pointing reason and infra-red chosen;
H06 does mention dynamo + battery and 17.8). So held-out ~ development (67-74%): no sign of overfitting.
Same failure types on unseen teams:
- memory error -> wrong role: H02 "Speaker C (Industrial Designer)"; memory has ES2013's B/C roles swapped
  (B = Kate = Industrial Designer, C = Steph = UI, per a #1/#262, c #763). The code role check enforced memory's wrong role.
- misses the key change: H04 never says the custom chip became a regular standard chip (the point of the question).
- proposal taken as done: H06 says they got under budget "by cutting the LCD and speaker"; those were only named as
  the most expensive items (d #459-461), the LCD stayed, and the target was reached by typing negative numbers (d #484-506).
- incomplete on 4-5 part questions (H03 no 16 Euros, H05 no D argument, H02 no 9.20 total).
Judge reliability: gap judge vs my reading 48% vs 71% here (vs ~4 points on the dev set) -> report both; user to spot-check.

## Phase 6c: Memory fixes from an outside review (deterministic, no LLM cost)

A second review (another chat, given PROBLEM_STATEMENT.md) found, and I confirmed on facts.db:
- 788 of 1,765 `updates` links point at a different subject; 97 facts "update" more than 3 others; proposals
  marked settled decisions as replaced. memory.py accepted every link.
- 23-35 of 125 role rows cite a turn with no role in it (ES2013 B/C: "you want an animal", "draw a cat");
  extract.py only checked the turn's speaker, and check_roles then enforced the wrong role.
- get_memory(topic) matched topic names only ("radio", "budget" found nothing).
Fixes:
1. memory.py: a link counts only if <= 3 ids, the old fact is earlier, shares a subject word, and a non-settled
   fact never replaces a settled one. Replaced facts 1,724 -> 1,502 (settled 554 -> 486).
2. memory.get_roles marks each role verified/unverified (role word in the cited turn or the 2 before, or the same
   speaker's next 2; project manager also by opening the kick-off). tools.get_roles shows verified only; deduction
   only from 3 verified. 109/125 verified; ES2013 B/C, TS3008 B/C, TS3003 B-D now letter only.
3. get_memory(topic) matches topic, summary or evidence, with L_C_D_ -> lcd.
New deterministic test tests/eval_current.py: 12 hand-checked final states (dev + held-out teams).
Before 6/12, after 11/12 (fix 3 alone 11/12, fix 1 alone 7/12). Remaining miss: ES2015 solar = extraction gap
(no solar facts at all). eval_memory unchanged 11/14, roles 4/4. Outcome check still fires only on T08.
Not done (future): run_plan adding memory chains + best hit per meeting; verifier call; checkable checklist.
Note: the held-out set shaped these fixes, so it is now a second dev set; no untouched teams are left for a new one.

**Runs after the memory fixes (gpt-4.1-mini):** dev judge 26/43 = 60%, 3/10; held-out judge 20/29 = 69%, 1/6.
My reading: dev ~29/43 = 67% (unchanged); held-out ~23/29 = 79% (up from ~71%: H05 now 5/5 because get_memory
"radio" finds the signal-type facts). The judge was wrong in BOTH directions this run: marked present facts missing
(T01 "speech recognition scrapped", T03 "not include the scroll-wheel", T08 "A decided against", H06 "hand dynamo
plus a battery") and a missing fact present (T09 "internet connection"). Judge numbers are not usable for small deltas.
T08 no longer states the LCD wrongly. H04 still misses the standard chip (5 tool calls, never called get_memory).
H06 again invents how they got under budget.
**Bug found: T04 accepted with 0 citations** - the model sent an EMPTY citations list 4 times with quotes inside
the answer; feedback only said "no citation passed". Fix: explicit "citations list is EMPTY" message, a check
against quoting inside the answer, and a last-resort salvage that verifies quotes found in the answer text
(exact or repaired >= 0.85, so it can't invent a citation). Dry-run tested.

**Model options checked (OpenAI pricing page, Nov 2026, $/1M in/out):** gpt-4.1-mini 0.40/1.60 (current);
gpt-6-luna 0.10/0.50 (released 22 Sep 2026, reasoning model, "most efficient model for focused, high-volume
tasks"; on Chat Completions function calling only works with reasoning_effort "none"); gpt-5.6-luna 0.20/1.20;
gpt-5-mini 0.25/2.00; gpt-5.4-nano 0.20/1.25; gpt-5.4-mini 0.75/4.50. agent.py/brief.py now have model_settings()
so AGENT_MODEL=gpt-6-luna works (reasoning_effort none via extra_body, no temperature). Default unchanged.
Laptop agent.py had been reverted again (lost the T04 fix) -> restored with model_settings added.

## Phase 7: Agent model switched to gpt-6-luna (reasoning_effort none, same code/prompts/memory)

- Dev (10): judge 35/43 = 81%, 5/10; my reading ~81%, 4-5/10 fully correct; 9/10 cite 2+ meetings;
  $0.049 for 10 (3.6x cheaper than gpt-4.1-mini), 13 s, 10.7 tool calls. Smoke test IS1004: both real reasons +
  exact final design, $0.0031 vs $0.014.
- Held-out (6): judge 19/29 = 66%, 0/6; my reading ~22/29 = 76% (H02 5/5: judge missed "too expensive ... could not
  make a double-curved case"). gpt-4.1-mini held-out ~79% -> about equal. Combined 72 facts: mini ~72%, luna ~79%.
- Fewer false claims: H06 now says the transcript doesn't show how they reached budget (no invented cuts);
  T08 no LCD claim. Still: H04 misses custom -> standard chip; T02 names B not A as objector; H03 skips meeting c.
- Catalist 25 (luna): 70 citations all valid, roles OK, 1-3 sentences. Q05 now right (battery+kinetic first ->
  battery only, A proposed skipping kinetic); Q07 now gives the display's 2-3 Euros (IS1008d) and the 12-12.5 total
  separately; Q25 "broccoli wins"; Q16 consistent. Weak citations in places (Q05 first quote, Q15 second).
- NEW format bug: luna writes its own citation markers into the answer text (hidden private-use chars
  "citeTS3006b" in Q01, "[[ES2004c]]" Q12, "[ES2008a]" Q20, "Citations: [..]" Q24, "[1][2][3]" H06).
  Fix (code): agent.clean_answer() strips them on every answer; run_questions.py --clean re-applies it to the
  existing answers.json without model calls (deterministic, same function).
- Brief (luna): 36 citations valid, none from d. Open disagreement now = missing finance info (the real one).
  Still lists C's rechargeable research as open although c ruled the rechargeable unit out.
