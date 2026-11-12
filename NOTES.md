# Build notes

A running log of what we built, what we decided and why. This becomes the
one-page write-up and the video script at the end.

## Assessment checklist (what Catalist is grading, and where we cover it)

| Criterion | Where we cover it | Status |
|---|---|---|
| Real agent behaviour (retrieve, loop, steps; holds at 10k meetings) | Phase 2 search, Phase 4 agent loop | todo |
| Memory over time (what is *current*; references back) | Phase 3 extraction + time-ordered events | todo |
| Tool-fit judgment (model vs code, defended) | Every phase: note it below as we go | in progress |
| Proof it works (10+ own questions, scored; cost + time for 1,000 meetings) | Phase 1b (first test questions), Phase 6 | todo |
| Honesty about failure (3 wrong answers, explained) | Phase 6 | todo |
| Platform reasoning (framework, models, APIs, where they broke) | Decided: plain Python + OpenAI SDK, SQLite. Log breakages as they happen | in progress |

### Model vs code log (add a line every phase)
- Parsing transcripts into turns: **code** (fixed format, regex is exact, free, repeatable)
- Token counting, stats, missing-meeting check: **code**
- Speaker-letter consistency check, keyword finder: **code** (fast regex scans; a human judges the results)
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
