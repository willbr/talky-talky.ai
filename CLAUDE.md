# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv**. Never invoke raw `python`, `pip`, or `venv` — always go through `uv run`.

```powershell
uv sync                                                # install deps
uv run flask --app app run                             # dev server on :5000
uv run python scripts/download_books.py                # populate data/books/
uv run python scripts/dump_db.py                       # print the inventory database as text
```

## Testing

No unit tests — the app is small enough that they cost more than they catch. Verify changes by running the app and exercising the feature in the browser. When doing that, point a second server at a scratch inventory so the real `inventory/` isn't touched: `INVENTORY_DIR` is relative to the working directory, so start it from an empty folder with `uv run --project <repo> flask --app <repo>/app.py run --port 5001`.

## Branching

Solo project, master-only. Commit and push directly to master — no feature branches, no worktrees.

## Architecture

Concatenative TTS: record short audio units, stitch them at synth time. Pre-deep-learning approach. No ML in the audio pipeline (yet). Future ML is trained from scratch on this project's own data with tinygrad — no pretrained models.

**Audio format is hard-coded everywhere: 16-bit mono PCM at 22050 Hz.** Reject any input that doesn't match. Constants live in `audio.py` (`SAMPLE_RATE`, `CHANNELS`, `SAMPLE_WIDTH`).

### Storage

- Metadata: SQLite at `inventory/talky.db` (stdlib `sqlite3`, no ORM). Audio: WAV files in `inventory/audio/<id>.wav` — never BLOBs.
- Tables: `units` (every recorded copy of a word or phoneme: `kind`, `label`, `source` = isolated/word/sentence, `source_unit_id`, `source_sentence_id`, `source_word`, `prev`, `next`, `position`, `duration_ms`), `sentences` (reference recordings), `alignment_segments` (empty until the sentence splitter lands), `schema_version`.
- IDs are `%Y%m%dT%H%M%S%f` UTC timestamps, unique across units and sentences, so lexicographic order is chronological. `inventory.is_valid_id()` guards any ID used in a path.
- Every recording **appends a copy**; nothing is overwritten. Rows whose WAV has been deleted by hand are skipped on read (not auto-deleted).
- `migrate.py` converts the pre-2026-11 file layout (`inventory/<word>.wav`, `inventory/phonemes/…`, `inventory/sentences/…`) the first time `talky.db` is created.

### Flask backend (Python)

- `app.py` — route handlers only; an app-factory `create_app()` registers all routes. Copies are addressed by `unit_id` (`/api/units…`, editor routes take `unit_id`, default most recent)
- `inventory.py` — `INVENTORY_DIR`, `normalise()` (lowercase, punctuation stripped), `is_valid_id()`
- `store.py` — all DB + audio-file access: `add_unit`, `get_unit`, `list_units`, `list_labels`, `delete_unit`, `replace_unit_audio`, `best_unit` (context scoring), sentences
- `migrate.py` — one-time legacy layout → DB import
- `phonemes.py` — canonical 39-symbol ARPABET set + CMUdict (`data/cmudict.dict`) grapheme-to-phoneme. Stress digits (`AH1`) are stripped at load time
- `audio.py` — DSP primitives: `trim_silence`, `peak_normalise`, `apply_fade`, `concatenate` (groups), `slice_samples`, `encode_wav`
- `extraction.py` — pure-math phoneme boundary proposal (RMS windowing + noise-floor threshold + ARPABET duration priors)
- `spectrogram.py` — numpy STFT → uint8 log-magnitude grid, base64'd into `/api/synthesize` and `/api/sentence/audio` responses for canvas rendering
- `synth.py` — `synthesise(sentence, phonemes_only=False)` tokenises with phrase positions, walks the resolver chain per token, falls back to per-phoneme via CMUdict. Returns `(wav, skipped, units)`; each unit carries ms spans and the `unit_id` of the copy used
- `resolver.py` — **the extensibility seam** for top-level tokens. `RESOLVERS = [WordResolver(), PhonemeResolver()]`, each `resolve(token, Context(prev, next, position))`. Add new fallback strategies by appending to this list
- `books.py` — sentence sampling from `data/books/*.txt` (Gutenberg corpus); filters to 30–250 char sentences

### Frontend (vanilla JS)

- `static/app.js` — single module, no framework. Two tabs: Train (record words + phoneme grid + word inventory) and Speak (synth + reference recording + book sample picker)
- Word and phoneme chips share one copies panel (`toggleUnitsPanel`): ▶ play, ✎ edit that copy, × delete
- `static/recorder-worklet.js` — AudioWorklet that captures Float32 audio from the mic
- WAV encoding is done client-side in `encodeWav()` (manual 44-byte header + int16 PCM). The server validates format
- Phoneme editor (in `app.js`) renders the waveform to canvas with draggable boundary handles; viewport state (`viewStartMs`/`viewEndMs`) handles zoom independently of absolute boundary positions

### Synthesis flow (read `synth.synthesise` if confused)

For each token, build a `Context` (previous/next word, position: `mid`, `comma` before `,;:`, `final` before `.!?` or at the end). Try every resolver; word copies score +1 per matching neighbour word and +1 for matching position, newest wins ties. If nothing resolves and the token isn't itself an ARPABET symbol, phonemise via CMUdict and pick each phoneme with `store.best_unit("phoneme", sym, prev=, next=)` (in-word neighbours). **All-or-nothing per word** — if any phoneme is missing, the whole word is skipped (returned in the `skipped` list, surfaced as red chips in the UI).
