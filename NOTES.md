# Build notes

A running log of what we built, what we decided and why. This becomes the
one-page write-up and the video script at the end.

## Phase 1: Data exploration (`src/explore.py`, plain code, no AI)

**Numbers**
- 138 meetings, 35 teams, ~677k words, ~880k tokens (avg ~6.4k tokens per meeting)
- 48,148 speaker turns. Median turn is **5 words**. **51% of turns are under 6 words**
- ~9% of all words are fillers (um, uh, yeah, okay)
- IS1002 has no meeting **a**; IS1005 has no meeting **d**

**Decisions that follow from the numbers**
1. **No "paste everything" approach.** 880k tokens doesn't fit in a prompt,
   and at 10,000 meetings it would be ~64M tokens. We retrieve instead.
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
5. **Cost baseline:** ~6.4M tokens to read 1,000 meetings once at ingest.

**Honest note for the interview:** one team's 4 meetings are only ~25k tokens,
which *would* fit in a prompt. We still retrieve, because in the real product
a team or company has hundreds of meetings, not four, and the brief explicitly
says the approach must hold at 10,000.
