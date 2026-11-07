# Diphone units — design

**Date:** 2026-11-07
**Project:** talky-talky.ai
**Status:** Implemented
**Builds on:** [2026-11-07-recording-splitter-design.md](2026-11-07-recording-splitter-design.md) (implemented)

## Why

Words with no word copy are built from whole phoneme copies, butt-joined
with 5 ms fades. Every join lands on a phoneme edge, which is exactly
where the transition cues that make speech intelligible sit, so they are
lost. A **diphone** runs from the middle of one phoneme to the middle of
the next. Joining diphones puts the cut in the steady middle of a sound
and keeps every transition whole. This is the classic step that made
concatenative synthesis intelligible.

The splitter already gives hand-checked phoneme boundaries for every
saved sentence, so diphones cost no extra recording.

## Goals

1. **Harvest on Save.** Saving a sentence in the splitter also writes one
   diphone copy per adjacent phoneme pair, crossing word edges, plus
   edge halves at pauses (`SIL-X`, `X-SIL`). These are replaced and
   excluded together with the sentence's other copies.
2. **Synthesise from diphones.** Runs of words without a word copy are
   built diphone by diphone and joined mid-phoneme with an aligned
   crossfade.
3. **Never worse than today.** A missing diphone is stood in for by
   halves of phoneme copies, so a word is only skipped when today's
   synth would also skip it.
4. **Pick for coverage.** The book picker gets a mode that chooses the
   passage adding the most missing diphones.

## Non-goals

- Prosody: no pitch or duration modification (no PSOLA).
- Diphones from isolated word recordings: they have no phoneme
  boundaries.
- A diphone browser in the UI. A count is enough for now.
- Changing the splitter's DTW reference builder, which stays on phoneme copies.

## Data

- **Schema v3.** `units.kind` allows `'diphone'`, and units gain `boundary_ms`,
  the offset of the A|B phoneme boundary inside a diphone copy. SQLite
  cannot change a CHECK constraint in place, so the upgrade rebuilds
  `units` (`store._upgrade_v2_to_v3`).
- **Label** `A-B`. `SIL` stands for a pause or the text's edge.
- **Context:**
  - `prev` is the phoneme before A and `next` the phoneme after B (None
    at a pause), so `best_unit` scores triphone-ish context.
  - `source_word` is the word the diphone sits in, or `w1 w2` when it
    crosses a word edge.

## Cuts (`diphones.cuts`)

- **Phrases.** A sentence's phonemes are split into phrases at
  punctuation, using the same rule as `phrase_context`, and around
  out-of-dictionary words, which have no phoneme segments.
- **Pairs.** Each phrase is `SIL p1 … pn SIL`, and each adjacent pair
  becomes one cut:
  - `A-B` runs from mid(A) − PAD to mid(B) + PAD. Its boundary is
    A.end; when the boundaries leave a gap, it is the gap's middle.
  - `SIL-X` runs from X.start to mid(X) + PAD.
  - `X-SIL` runs from mid(X) − PAD to X.end.
- **PAD** = 12 ms (my call). It is overlap material for the crossfade,
  clamped to the recording.
- **Minimum length:** 20 ms (my call; phonemes keep 30 ms).

## Synthesis

1. **Resolve each token.**
   - A word copy, or a raw ARPABET token, works as today: one faded
     piece. `phonemes_only` drops word copies, as today.
   - Otherwise the word's CMUdict pronunciation joins the current
     **run**.
2. **Runs end** at a word copy, at a skipped token, or after a
   comma/final token.
3. **Pauses.** After a comma token there is 150 ms of silence, and after
   a final token that isn't last there is 300 ms (my call; today there is
   none).
4. **Filling a slot.** A run is `SIL p1 … pn SIL`. Each adjacent pair is
   filled by:
   - the best diphone copy (`best_unit("diphone", "A-B", prev, next)`); or
   - **fallback:** the right half of A's phoneme copy plus the left half
     of B's (SIL contributes nothing). Each phoneme occurrence uses one
     copy for both of its halves, so an all-fallback run sounds like
     today's synth with the join moved to the middle of the sound.
5. **Joins inside a run.**
   - Overlap 15 ms with linear ramps.
   - First slide the incoming piece up to 8 ms to the offset with the
     best normalised correlation against the outgoing tail, so periodic
     sounds line up in phase.
   - Joins between groups (word copies, runs, pauses) keep the 5 ms fades.
6. **Skips.** If a slot can be filled neither way, the word owning the
   missing phoneme is skipped and the run splits around it.
7. **Response.** It carries token spans and phoneme spans, as today
   (each phoneme spans from its slot boundary to the next one), plus
   `diphones: {real, fallback}` counts. Speak shows these as a line under
   the output.

## Coverage picker

- **Picking.** `length: "coverage"` samples 200 candidate passages,
  using the same passage rules, from the chosen book (or any), and
  prefers those of 150–500 characters. Each is scored by the
  frequency-weighted count of diphones it contains that the inventory
  lacks (weights from the candidate pool), divided by the square root of
  its phoneme count: a plain sum always picked the longest passage, and
  a plain ratio picked one-liners. The best one is returned with
  `new_diphones` and `have` counts.
- **Book cache.** Book sentences are cached per process so that 200
  samples are cheap.
- **UI:**
  - A `passage, fills gaps` option in the length dropdown.
  - The pick's status line says how many new diphones the passage
    adds.
  - The Train status strip shows the diphone count.

## Verification

Run on a scratch copy on :5001: upgrade the copied DB, re-save a
sentence, check the diphone rows (`scripts/dump_db.py`), synthesise and
listen/inspect, and pick for coverage. Back up the real inventory before
the first v3 open.
