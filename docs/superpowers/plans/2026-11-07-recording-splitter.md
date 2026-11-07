# Recording Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record passages of any length on the Train tab. Propose sentence, word and phoneme boundaries with DTW against a synthesised reference, let the user correct them one sentence at a time, and harvest context-rich word and phoneme copies on save.

**Architecture:**
- `align.py` (new) is the pure proposer: MFCC features, DTW with pins, sentence splitting, reference synthesis.
- `store.py` gains schema v2: recordings, alignment segments with a sentence tier, status and edited flags, and per-sentence copy replacement.
- `synth.py` gains a shared `phrase_context()` (cross-word phoneme neighbours, with context reset at punctuation) used by both synth and harvest.
- `app.py` gains the recording/splitter endpoints and harvest.
- `static/splitter.js` (new ES module) is the editor, imported by `app.js`.

**Tech Stack:** Python 3.13, numpy, Flask 3, stdlib sqlite3, vanilla JS (ES modules, canvas).

**Spec:** `docs/superpowers/specs/2026-11-07-recording-splitter-design.md`

## Global Constraints

- 16-bit mono PCM at 22050 Hz everywhere. No new dependencies. Run everything with `uv run`.
- **No unit tests** (user decision). Every task is verified manually against a **scratch inventory**, never the real `inventory/`:
  - Start a server from an empty scratch dir: `uv run --project <repo> flask --app <repo>/app.py run --port 5001`.
  - Run Python checks with the working directory set to the scratch dir, because `INVENTORY_DIR` is relative.
- Proposal speed is about 100 ms per sentence in Python.
- No pretrained models. Recording WAVs are never cut into files; segments are time spans.
- Copies from a recording: one gain per recording (peak of the whole take → `audio.PEAK_TARGET`), no per-copy trim or normalise, and a 30 ms minimum slice.
- The pause rule follows the text: context (word and phoneme prev/next) never crosses `, ; : . ! ?`.
- Commit directly to master. Back up the real `inventory/` before the first v2 code touches it.

## Review Focus

1. **The real DB is upgraded from v1 to v2 in place** (2 sentences become recordings, 127 units keep their data, and `source_sentence_id` is renamed). Task 1 verifies this on a *copy* of the real DB before anything touches the original.
2. **Re-saving a sentence** must replace its copies, never add to them. Task 4 checks that the copy count is stable across two saves and that the old WAVs are gone.
3. **A misread passage.** Excluding a sentence deletes its copies, and editing a sentence's text re-proposes just that sentence (Task 4/5 manual check).
4. **The sentence split is wrong** (e.g. a pause inside a sentence). Dragging the first or last word outside the proposed span must still work: the pins widen the span (Task 3 check with a pin outside the span).
5. **Sentence-sourced copies opened from the copies panel or a Speak chip** must open the splitter, not the trim editor. The trim endpoints return 409 (Task 4 curl and Task 5 browser check).

---

### Task 1: Schema v2 and store API

**Files:** Modify `store.py`, `migrate.py`.

**Interfaces — produces (in `store.py`):**
- `SCHEMA_VERSION = 2`. `create_schema(conn)` builds v2 on a fresh DB and upgrades v1 in place:
  - `ALTER TABLE sentences RENAME TO recordings`
  - `ALTER TABLE units RENAME COLUMN source_sentence_id TO source_recording_id`
  - `ALTER TABLE units ADD COLUMN source_sentence_idx INTEGER`
  - drop and recreate `alignment_segments`
  - set the version to 2
- New `alignment_segments`:

  ```sql
  CREATE TABLE alignment_segments (
    recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    tier TEXT NOT NULL CHECK (tier IN ('sentence','word','phoneme')),
    sentence_idx INTEGER NOT NULL,   -- owning sentence (= idx for sentence rows)
    idx INTEGER NOT NULL,            -- index within the tier, within the sentence (sentence rows: sentence index)
    label TEXT NOT NULL,             -- sentence text / normalised word / ARPABET symbol
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    parent_idx INTEGER,              -- phoneme → word idx; word → sentence idx
    edited INTEGER NOT NULL DEFAULT 0,
    status TEXT CHECK (status IN ('proposed','saved','excluded')),  -- sentence rows only
    PRIMARY KEY (recording_id, tier, sentence_idx, idx)
  );
  ```

- `UNIT_FIELDS` gains `source_recording_id` and `source_sentence_idx` (replacing `source_sentence_id`). `add_unit(..., source_recording_id=None, source_sentence_idx=None, ...)`.
- `best_unit(kind, label, *, prev=None, next=None, position=None, exclude=None)`, where `exclude: Callable[[dict], bool] | None` drops candidate copies.
- Recordings:
  - `save_recording(text, wav_bytes) -> str` (replaces `save_sentence`)
  - `recording_path(rec_id) -> Path`
  - `get_recording(rec_id) -> dict | None` (`{id, text, duration_ms}`)
  - `list_recordings() -> list[dict]`
  - `find_recordings(text) -> list[dict]`: `[{recording_id, sentence_idx|None}]`, newest first. It matches the whole normalised text, or the normalised label of any sentence-tier row.
  - `find_sentence_by_text` is removed; Speak uses `find_recordings`.
- Sentences:
  - `set_sentences(rec_id, rows: list[dict])` inserts sentence-tier rows `{idx, label, start_ms, end_ms}` with status `proposed`.
  - `get_sentences(rec_id) -> list[dict]` (`{idx, text, start_ms, end_ms, status}`)
  - `update_sentence(rec_id, idx, *, text=None, start_ms=None, end_ms=None, status=None)`
- Segments: `get_segments(rec_id, sentence_idx) -> {"words": [...], "phonemes": [...]}`. Rows are `{idx, label, start_ms, end_ms, parent_idx, edited}`.
- `replace_sentence(rec_id, sentence_idx, *, words, phonemes, units: list[dict]) -> list[str]`:
  - In one transaction: delete the sentence's word/phoneme segment rows and the units with that `(source_recording_id, source_sentence_idx)`; insert the new segments; insert the unit rows (their WAVs were already written by the caller via `write_unit_wav`); set status `saved`.
  - After commit, unlink the old units' WAVs. Returns the new unit ids.
- `clear_sentence_copies(rec_id, sentence_idx)` deletes those units and their WAVs (used by exclude).
- `write_unit_wav(wav_bytes) -> str` mints an id and writes `audio/<id>.wav` (used by `add_unit` and the harvest).
- `migrate.py` inserts into `recordings`/`source_recording_id`.

- [ ] **Step 1:** Back up the real inventory to the scratchpad (`inventory-backup-pre-v2`).
- [ ] **Step 2:** Implement the above.
- [ ] **Step 3: Verify the v1→v2 upgrade on a copy.** Copy the real `inventory/` into a scratch dir, cd there, and run `uv run --project <repo> python <repo>/scripts/dump_db.py`. Expected:
  - 2 recordings, 86 words, 39 phonemes, no errors;
  - `PRAGMA user_version`/`schema_version` is 2;
  - `units` has `source_recording_id` and `source_sentence_idx`.
- [ ] **Step 4: Verify a fresh legacy migration.** Point a scratch dir at a copy of the *pre-SQLite* backup (`inventory-backup-20261107`) and run `dump_db.py`. Expected: the counts from the earlier migration (86/39/2).
- [ ] **Step 5:** Update `scripts/dump_db.py` (recordings with sentence-tier rows and status). Commit: "Store: schema v2 — recordings, sentence tier, per-sentence copy replacement".

### Task 2: Phrase context in synth (cross-word, pause rule)

**Files:** Modify `synth.py`, `resolver.py` (unchanged signature).

**Interfaces — produces:**
- `synth.TokenCtx` dataclass:
  - `token: str`
  - `position: str`
  - `prev_word: str | None`
  - `next_word: str | None`
  - `phonemes: list[str] | None` (CMUdict or None)
  - `prev_phoneme: str | None` (neighbour of the first phoneme)
  - `next_phoneme: str | None` (neighbour of the last phoneme)
- `synth.phrase_context(text: str) -> list[TokenCtx]`:
  - Builds on `tokenise_with_positions`.
  - `prev_word` and `prev_phoneme` are None when the previous token's position is not `mid` (a punctuation pause sits between them), or when this is the first token.
  - `next_word` and `next_phoneme` are None when this token's position is not `mid`, or when this is the last token.
  - `prev_phoneme`/`next_phoneme` come from the neighbour's CMUdict pronunciation (last or first symbol), and are None if the neighbour is out of dictionary.
- `synthesise()` uses `phrase_context`:
  - Word resolver `Context(prev=prev_word, next=next_word, position=position)`.
  - Phoneme j of a word uses `prev = phs[j-1]` or `prev_phoneme` at j=0, and `next = phs[j+1]` or `next_phoneme` at the end.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify** in a scratch dir with a seeded store.
  - `uv run python -c "import synth; print(synth.phrase_context('Alice thought, she wait.'))"` shows:
    - alice: `next_word=thought`, `next_phoneme=TH`;
    - thought: `next_word=None` (comma);
    - she: `prev_word=None`.
  - Synthesising a phrase with a seeded `S` copy whose `next=TH` picks that copy for "alice thought". Check via `units[0]["phonemes"][-1]["unit_id"]`.
- [ ] **Step 3:** Commit: "Synth: cross-word phoneme context; context resets at punctuation".

### Task 3: `align.py` — the proposer

**Files:** Create `align.py`.

**Interfaces — produces:**
- `split_text_sentences(text: str) -> list[str]`: splits at `. ! ?` followed by whitespace, drops empties, and keeps the punctuation.
- `propose_sentence_spans(samples: np.ndarray, sentences: list[str]) -> list[tuple[int, int]]`:
  - One sentence → the loud region (`extraction._loud_region_ms`).
  - Several → pauses are runs of quiet 20 ms RMS windows ≥150 ms inside the loud region (quiet threshold as in `extraction`). Expected break times come from cumulative phoneme-duration priors (out-of-dictionary words: `max(150, 60·len(word))` ms) scaled to the loud span. Each break picks the nearest unused pause after the previous one, falling back to the expected time. A sentence ends at `pause.start`, and the next starts at `pause.end`.
- `build_reference(sentence_text: str, exclude=None) -> Reference`. `Reference` is a dataclass:
  - `samples: np.ndarray`
  - `words: list[dict]` (`{label, start_ms, end_ms, in_dict}`)
  - `phonemes: list[dict]` (`{label, start_ms, end_ms, word_idx}`)

  Built via `synth.phrase_context` with cross-word phoneme picks through `store.best_unit(..., exclude=exclude)`:
  - Missing phoneme → a silent placeholder of its `extraction.PHONEME_DURATION_MS`.
  - Out-of-dictionary word → silence of `max(150, 60·len)` ms.
  - 150 ms of silence after `comma` tokens, and 100 ms of silence padding at both ends.
  - Copies are concatenated with `audio.apply_fade`.
- `features(samples: np.ndarray) -> np.ndarray`, shape (frames, 14):
  - 512-point FFT, hop **220 samples (~10 ms)**, Hann window.
  - 40 mel bands (0–8000 Hz); log; DCT-II; keep c1–c13; add normalised log-energy as the 14th column.
  - Per-recording mean/variance normalisation.
- `dtw_path(a: np.ndarray, b: np.ndarray) -> np.ndarray`, an (L, 2) array of (i, j) pairs from (0, 0) to (N-1, M-1):
  - Euclidean cost; anti-diagonal vectorised accumulation; argmin back-pointers stored as int8.
  - The algorithm the signature doesn't determine:
    ```
    D[i,j] = C[i,j] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
    process diagonals k = i+j in increasing order; each diagonal is a vector op over its valid i
    ```
- `propose(rec_samples, span_ms: tuple[int,int], ref: Reference, pins: list[dict] = ()) -> dict`, returning `{"words": [...], "phonemes": [...], "span_ms": [s, e]}` in absolute recording ms.
  - The recording span is padded by 100 ms each side (clamped).
  - **Pins** are `{"tier": "word"|"phoneme", "idx": int, "edge": "start"|"end", "ms": int}`. Each one maps the matching reference edge time to `ms`. The span is widened to include every pin, and DTW runs independently between consecutive anchor pairs: span start, pins sorted by ref time, span end.
  - A reference edge maps to the recording frame at the first path point whose ref frame is ≥ the edge frame.
  - Post-process: monotonic, ≥5 ms per segment, word edges = their first/last phoneme edges (out-of-dictionary words map their placeholder edges), pinned edges exact.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on real audio** in a scratch copy of the real inventory (after Task 1). A one-off script (scratchpad, not committed) prints the proposal for the Alice recording and times it.
  - Expected: 25 words in order, start times monotonic, total runtime ≤ ~300 ms, and plausible edges.
  - Spot-check against the known Synth spectrogram: "all" around the long pause region.
  - Repeat with one pin outside the proposed span; the span widens and that edge equals the pin.
- [ ] **Step 3:** Commit: "Align: DTW proposer against a synthesised reference, with pins".

### Task 4: Endpoints, harvest, books passages, Speak lookup

**Files:** Modify `app.py`, `books.py`, `scripts/dump_db.py` (if not done).

**Interfaces — produces (HTTP):**

| Route | Request | Response |
|---|---|---|
| `POST /api/recordings` | form `text`, `audio` | Validates the WAV and saves it raw. Splits text into sentences and stores sentence rows (`propose_sentence_spans`). → `{id, sentences: [...]}` |
| `GET /api/recordings/find?text=` | — | `{matches: [{recording_id, sentence_idx}]}` newest first |
| `GET /api/recordings/<id>` | — | `{id, text, duration_ms, sentences: [{idx, text, start_ms, end_ms, status}], takes: [ids of recordings with the same normalised text, oldest first]}` |
| `GET /api/recordings/<id>/audio` | — | `audio/wav` (absolute path) |
| `POST /api/recordings/<id>/sentences/<idx>/propose` | `{pins?, text?, fresh?}` | Saved + no pins/text/fresh → the stored alignment. Otherwise a new proposal (`text` updates the sentence row first; the reference excludes this sentence's own copies). → `{sentence, words, phonemes, view_ms: [s, e], spectrogram}` (spectrogram of `view_ms`, which is the span ± 300 ms) |
| `POST /api/recordings/<id>/sentences/<idx>/save` | `{words, phonemes}` each with `edited` | Harvest (below), then `replace_sentence`. → `{saved_units: N, sentence}` |
| `POST /api/recordings/<id>/sentences/<idx>/exclude` | `{excluded: bool}` | Status `excluded` + `clear_sentence_copies`, or back to `proposed` |
| `POST /api/books/sample` | `{book, length: "sentence"\|"passage"}` | passage = `books.sample_passage` |
| `POST /api/sentence/audio` | unchanged | Matches via `find_recordings`. An inner sentence returns that span's audio (sliced, re-encoded) and spectrogram |

Harvest (in `app.py`, a helper `_harvest(rec_id, idx, words, phonemes) -> list[dict]`):
- Gain = `PEAK_TARGET / peak(whole recording)`.
- Contexts come from `synth.phrase_context(sentence text)`, matched to `words` by index (the counts must be equal, else 400).
- Each word span ≥30 ms → a word copy (`source="sentence"`, prev/next word, position).
- Each phoneme span ≥30 ms → a phoneme copy with cross-word prev/next.
- Excluded or out-of-dictionary words still get word copies; out-of-dictionary words have no phonemes.

Guards:
- `/api/extract/propose|save|trim` and `/api/phoneme/propose|trim` targeting a `source=sentence` unit → 409 `{error: "sentence copy", recording_id, sentence_idx}`.
- Synth units gain `source_recording_id` and `source_sentence_idx`.
- `/api/sentence/record` and `/api/sentences` are removed.

`books.sample_passage(book=None, n_min=3, n_max=5) -> tuple[str, str] | None`: consecutive sentences from the unfiltered split, each 2–250 characters.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify with curl** against :5001 on a scratch copy of the real inventory.
  - Post the Alice WAV (from the backup) as a new recording → 1 sentence.
  - Propose → 25 words.
  - Save → copies created; `dump_db.py` shows `sentence` copies with cross-word context.
  - **Save again** → copy count unchanged, old WAVs gone.
  - Exclude → copies gone.
  - Trim endpoint on a sentence copy → 409.
  - Books passage returns 3–5 sentences.
  - `/api/sentence/audio` finds the recording.
- [ ] **Step 3:** Commit: "API: recordings, splitter propose/save/exclude, harvest, passages".

### Task 5: Frontend — Record panel and splitter editor

**Files:** Modify `templates/index.html`, `static/app.js`, `static/style.css`. Create `static/splitter.js`.

- **Train → Record panel.**
  - Title "Record"; a multi-line `#record-text` (replaces `#word-input`); book picker + length select (`sentence`/`passage`) moved here.
  - Record/Save/Discard; status line; a `#take-hint` button "Open recording (N takes)" shown when `/api/recordings/find` matches the text (debounced 400 ms on input, and after Pick).
  - Save: armed phoneme → as today; one normalised word → `saveWord`; otherwise `POST /api/recordings`, then open the splitter.
- **Speak.** Remove the reference panel and the book picker. A red skipped chip → Train with the word in `#record-text`.
- **`splitter.js`** exports `initSplitter(deps)` and `openSplitter(recordingId, sentenceIdx = 0)`. `deps = {mount: Element, setStatus, refreshInventory}`. The editor mounts after the Record panel.
  - Header: "Take t/T · Sentence i/n · status", take switcher, ◀ ▶, ×.
  - Editable sentence text input, which re-proposes on change (debounced, unsaved sentences).
  - One canvas: the spectrogram of `view_ms` (top), a word band, a phoneme band.
  - Word edges drag both bands' coincident edges. Phoneme edges are clamped inside their word. Pauses (gaps) are grey. Pinned edges are drawn in the accent colour.
  - Mouse-up after a drag adds the pin `{tier, idx, edge, ms}` and posts propose with all pins.
  - Click a word or phoneme cell to play it, via an AudioBuffer of the recording, decoded once.
  - Buttons: ▶ sentence, Re-propose (clear pins, `fresh: true`), Exclude/Include, Save sentence. Save marks edited segments: any segment with a pinned or dragged edge.
  - Progress "k/n saved".
- **Copies panel and Speak chips.** A unit with `source === 'sentence'` → `openSplitter(source_recording_id, source_sentence_idx)` (on Train).

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify in the browser** (Playwright) on the :5001 scratch copy:
  - Load the Alice text → take hint → open the splitter → proposal drawn.
  - Drag + release a word edge → re-proposed, pin shown.
  - Save → progress 1/1, copies in the word panel with "from sentence" metadata.
  - ✎ on that copy opens the splitter.
  - Pick a passage → 3–5 sentences in the box.
  - Speak synth shows the reference.
  - No console errors other than the favicon.
- [ ] **Step 3:** Commit: "UI: Record panel for passages + sentence splitter editor".

### Task 6: Real data, docs, push

- [ ] **Step 1:** Restart :5000 on the real inventory (backed up in Task 1); the v2 upgrade runs. `dump_db.py` shows 2 recordings, 86 words, 39 phonemes.
- [ ] **Step 2:** Update `CLAUDE.md` (storage v2, `align.py`, `splitter.js`, recordings flow). Set the spec status to Implemented.
- [ ] **Step 3:** Commit and push.
