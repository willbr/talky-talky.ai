# Recording splitter — design

**Date:** 2026-11-07
**Project:** talky-talky.ai
**Status:** Implemented
**Builds on:** [2026-11-07-sqlite-unit-store-design.md](2026-11-07-sqlite-unit-store-design.md) (implemented)

## Why

The synth is unintelligible. Every copy in the library was recorded in
isolation: a word said on its own, or a phoneme cut from one. Real
speech blends sounds across neighbours, and listeners rely on those
transitions. The unit store can now hold many copies and choose by
context, but it has nothing context-rich to choose from.

The user will record **full passages**: a sentence, a paragraph or
longer, in one take. The splitter turns each take into word and phoneme
copies that carry their real neighbours and phrase position. The
hand-corrected boundaries are kept as training data. A future neural
boundary proposer will be trained **from scratch on this data with
tinygrad** (no pretrained models, ever), so each corrected recording
makes the next proposal better.

## Goals

1. Recording passages of any length becomes the main way to record,
   on the Train tab.
2. For each recording, the app proposes sentence, word and phoneme
   boundaries automatically. The user corrects them in an editor.
3. Saving a sentence's alignment stores the boundaries and regenerates
   that sentence's word and phoneme copies, with context.
4. Synth chooses phoneme copies by **cross-word** neighbours (the user's
   decision; it is the triphone idea from speech recognition).
5. Work on long recordings can be spread across sittings: progress is
   per sentence.

## Non-goals

- Any neural network or tinygrad code. That's the next project. This one
  only makes sure the alignments are a clean dataset for it.
- Live re-alignment while dragging (60 fps). That's tinygrad/GPU work
  later. Here, re-alignment runs when a drag is released.
- Changing how copies are scored beyond the cross-word phoneme context
  and the pause rule below.
- Crossfades, pitch/energy smoothing, or other concat-quality work.
- Unit tests. The repo deliberately has none; verify by running the app.

## Anti-goals

- Any pretrained model or third-party aligner (MFA, wav2vec, …).
- Cutting the recording's audio into separate files for sentences. A
  take stays one WAV; segments are time spans over it. (Word and phoneme
  *copies* are still written as their own WAVs, as today.)
- A save that silently duplicates copies. Re-saving a sentence must
  replace that sentence's copies, never add to them.
- An editor where you must place hundreds of boundaries in one view.

## Constraints

- Audio: 16-bit mono PCM, 22050 Hz, as everywhere.
- Python + numpy on the server; stdlib `sqlite3`; uv only; no new
  dependencies.
- Proposal speed: **about 100 ms per sentence** in Python is acceptable
  (user's figure). Whole-recording proposal time scales with sentence
  count.
- Recordings live in the existing store: metadata in `inventory/talky.db`,
  audio in `inventory/audio/<id>.wav`.
- Commit directly to master.

## Decisions

### Recording (Train tab)

- The Train tab's main panel becomes **Record**. It has a text box, the
  book picker (moved here from Speak), and Record / Save / Discard.
- If the text is a **single word**, it is saved exactly as today: a word
  copy, followed by the existing word editor for splitting it into
  phonemes. *(Designer's call: one box, two paths.)*
- If the text is **more than one word**, it is saved as a recording and
  the splitter editor opens under the panel with proposals already drawn.
- Recordings are stored raw (no trim or normalise), as reference
  sentences are today.
- The phoneme grid's isolated recording stays as is.
- Speak loses its "Reference recording" panel. A red skipped chip on
  Speak still switches to Train with that word typed in, ready to record.

### Data

- The existing `sentences` table becomes the table of **recordings**: one
  take of any length with its text. *Designer's call:* rename it to
  `recordings` in a hand-written schema v2 migration, along with
  `units.source_sentence_id` → `source_recording_id`. The builder may
  keep the old names if renaming proves awkward, but the meaning
  changes either way.
- Units gain `source_sentence_idx`, the sentence index within the
  recording. That makes "this sentence's copies" a precise set to
  replace on save.
- `alignment_segments` gains a third tier, `sentence`. Rows:
  `(recording_id, tier, idx, label, start_ms, end_ms, parent_idx)`.
  - Phoneme rows point at their word.
  - Word rows point at their sentence.
  - Shaped like Praat TextGrid tiers.
  - The table is empty today, so v2 can recreate it.
- Each sentence has a **status**: `proposed` (machine boundaries only)
  or `saved` (user corrected and saved). *Designer's call:* store it on
  the sentence-tier row, or in a small side table. Only `saved`
  sentences produce copies and count as training data.
- The 2 existing reference recordings become recordings waiting to be
  split. No re-recording is needed.

### Proposing boundaries

All server-side Python, written as a **pure function**: recording
samples + text in, segment tiers out. A tinygrad version can then
replace it later without touching storage or the UI.

1. **Sentences.**
   - Split the text into sentences at `. ! ?`.
   - Find pauses in the audio (low-energy runs, roughly ≥150 ms, using
     the noise-floor approach already in `extraction.py`).
   - Assign the N−1 sentence breaks to the pauses that best match each
     break's expected time. Expected times come from cumulative
     phoneme-duration priors scaled to the recording length.
   - A single-sentence recording skips this step.
2. **Words and phonemes within each sentence** (DTW against synth):
   - Synthesise the sentence text in phonemes-only mode. Its word and
     phoneme spans are known exactly.
   - Insert short silences at commas, so commas line up with the user's
     pauses.
   - For words not in CMUdict, insert a silent placeholder whose length
     is a duration guess.
   - Compute MFCC-style frames for both (per-recording mean
     normalisation).
   - Run DTW, and map every synth boundary through the warp path onto
     the recording.
   - Words not in CMUdict get a word span (their placeholder's mapped
     span) and no phoneme spans.
3. **Pinned re-proposal.** When the user releases a dragged boundary, it
   becomes pinned. The sentence re-proposes with DTW forced through the
   pins (equivalently, DTW run separately between consecutive pins), so
   pinned edges never move.

Speed target: about 100 ms per sentence. Boundary resolution should be
fine enough for short consonants (≤10 ms is the guide); hop size and
feature details are the builder's call.

Pinning a phoneme phone (the case of a synthesis gap) is left to the
builder.

### Editor (Train tab, under Record)

- Shows **one sentence at a time**, zoomed into the long recording, with
  ◀ ▶ to step between sentences and a "3/8 saved" progress count.
- Spectrogram on top. Under it, a **word row** and a **phoneme row**,
  with the sentence's own edges also draggable. This is the layout from
  the earlier mockup.
  - Word edges are shared by both rows.
  - Phoneme edges stay inside their word.
  - Pauses show as grey gaps and are never saved as copies.
- Click any word or phoneme to hear it.
- Releasing a drag pins that edge and re-proposes the rest of the
  sentence.
- **Re-propose** discards the sentence's unsaved edits and pins and
  proposes fresh. That's useful after the library has improved.
- **Save sentence** stores its boundaries, marks it `saved`, and
  regenerates its copies.
- Reuses the existing editor's zoom/viewport and drag code where it fits.
- Opening a recording that is already partly split shows saved sentences
  as saved, and proposes the rest.
- **Finding older recordings** (designer's call): load the text into the
  Record box (type it or pick it from the book). The app finds the
  matching recording and reopens its editor. No separate list UI.

### Copies produced on save

For each word in the saved sentence:
- **Word copy.** `source="sentence"`, `source_recording_id`,
  `source_sentence_idx`, prev/next word, and position from the
  punctuation rule synth already uses.
- **Phoneme copies.** One per phoneme, with **cross-word** prev/next
  phonemes.

The pause rule (designer's call, applies to both synth and harvest):
- Context does not cross a comma, sentence end or other punctuation
  pause. There, prev/next is null, because silence separates the sounds.
- This applies to word neighbours too, which slightly changes today's
  synth scoring across commas.

On save, delete every unit with that `source_recording_id` +
`source_sentence_idx` (and their WAVs), then insert the new ones, in one
transaction.

### Synth changes

- Phoneme slots use **cross-word** neighbours. For the first phoneme of
  a word, prev is the last phoneme of the previous word's CMUdict
  pronunciation, and next works the same way at the end. That holds even
  if the neighbouring word was rendered from a word copy, as long as no
  pause punctuation sits between them.
- Word context also resets at pause punctuation.
- Existing single-word phoneme copies (null at their edges) still work.
  They just lose the edge point where a neighbour exists.

### Speak: reference lookup

When the typed text matches a whole recording or any **saved or
proposed sentence** inside one, Speak shows that span's audio and
spectrogram. For an inner sentence, the audio is sliced to its span.

## Settled after the builder check (designer's calls; user said "build it")

1. **Audio that doesn't match the text** (misreads, skips, repeats):
   - Sentence-tier rows are created when the recording is first
     proposed, and each one stores **its own text**. Sentence indices are
     fixed from then on.
   - In the editor, an unsaved sentence's text can be edited. That
     re-proposes only that sentence.
   - A sentence can be marked **excluded**: it produces no copies and is
     not training data.
   - Status values: `proposed` | `saved` | `excluded`. Saved sentences'
     text is locked; un-save by re-proposing.
2. **Editing sentence-sourced copies.**
   - The trim/split endpoints refuse `source=sentence` copies with a 409.
   - In the copies panel, ✎ on a sentence copy, and the Speak edit chip
     for one, open the **splitter at that recording and sentence**
     instead. That way a re-save can never orphan or overwrite
     hand-edits.
3. **Loudness.** One gain per recording: peak-normalise the whole take
   once (to `audio.PEAK_TARGET`) and cut copies from that. Relative
   loudness inside the take is preserved, so quiet consonants stay quiet.
   No per-copy trim or normalise.
4. **The pause rule follows the text** (punctuation), so harvest stays
   consistent with synth, which only has text to go on.
5. **Book picker passages.**
   - The picker gains a length choice: *sentence* (today), or *passage*
     (3–5 consecutive sentences from the same book, via a new
     `books.sample_passage`).
   - The Record text box is multi-line.
6. **The same text recorded twice.**
   - Every Record makes a new take.
   - Loading text opens the newest matching take. A small take switcher
     ("take 1 / 2") in the editor header reaches older ones.
   - Text that matches one sentence inside a longer take opens that
     take at that sentence.
   - Speak shows the newest match.
7. **Human vs machine boundaries.** Each alignment segment row gets an
   `edited` flag: 1 if the user dragged or pinned either of its edges
   before saving. It's added in the v2 migration, so the future trainer
   can weight hand-placed edges.

Also decided:
- Proposals for a sentence exclude that sentence's own copies when
  synthesising the DTW reference.
- In-dictionary words with missing phoneme copies get a silent
  placeholder (like out-of-dictionary words).
- Proposals are recomputed, not stored, until a sentence is saved.
  Sentence-tier rows are stored at creation, as above.

## Left to the builder

- MFCC details, hop size, DTW variant (banding, step pattern), pause
  thresholds, and the silence/placeholder durations.
- The endpoint shapes for: create recording, propose (whole recording or
  one sentence with pins), save sentence, get recording with alignment.
- Schema v2 migration mechanics (renames, recreating
  `alignment_segments`).
- How the editor code is shared with the existing word/phoneme editor.
- Manual verification plan. Use a scratch-inventory server on :5001 (see
  CLAUDE.md "Testing"). At minimum:
  - record a 2–3 sentence passage, and check proposals look plausible;
  - drag + release to pin;
  - save one sentence and check its copies in the panels and in
    `scripts/dump_db.py`;
  - re-save it, and confirm there are no duplicate copies;
  - synth a phrase and check it uses the new copies;
  - check Speak shows the inner-sentence reference.
- Updating CLAUDE.md and `scripts/dump_db.py` for recordings and
  alignments.
