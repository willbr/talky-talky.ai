# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv**. Never invoke raw `python`, `pip`, or `venv` — always go through `uv run`.

```powershell
uv sync --group dev                                    # install deps
uv run flask --app app run                             # dev server on :5000
uv run pytest                                          # full test suite
uv run pytest tests/test_app.py::test_record_saves_valid_wav -v   # single test
uv run python scripts/download_books.py                # populate data/books/
```

## Branching

Solo project, master-only. Commit and push directly to master — no feature branches, no worktrees.

## Architecture

Concatenative TTS: record short audio units, stitch them at synth time. Pre-deep-learning approach. No ML in the audio pipeline (yet).

**Audio format is hard-coded everywhere: 16-bit mono PCM at 22050 Hz.** Reject any input that doesn't match. Constants live in `audio.py` (`SAMPLE_RATE`, `CHANNELS`, `SAMPLE_WIDTH`).

### Flask backend (Python)

- `app.py` — route handlers only; an app-factory `create_app()` registers all routes
- `inventory.py` — disk layout for recorded audio
  - words: `inventory/<normalised>.wav` (lowercase, punctuation stripped via `normalise()`)
  - phonemes: `inventory/phonemes/<lowercase-symbol>/<id>.{wav,json}` — every save appends a *variant*; the JSON sidecar holds `{id, symbol, source, prev, next, duration_ms}` (`source` is `"isolated"` or the word it was extracted from). IDs are `%Y%m%dT%H%M%S%f` UTC timestamps, so lexicographic order is chronological; `is_valid_variant_id()` guards any ID used in a path
  - legacy flat `inventory/phonemes/<symbol>.wav` files are lazily migrated to `<symbol>/00000000T000000000000.wav` on first `list_variants()`
  - reference sentences: `inventory/sentences/<utc-timestamp>.{wav,txt}` (paired)
- `phonemes.py` — canonical 39-symbol ARPABET set + CMUdict (`data/cmudict.dict`) grapheme-to-phoneme. Stress digits (`AH1`) are stripped at load time
- `audio.py` — DSP primitives: `trim_silence`, `peak_normalise`, `apply_fade`, `concatenate` (groups), `slice_samples`
- `extraction.py` — pure-math phoneme boundary proposal (RMS windowing + noise-floor threshold + ARPABET duration priors)
- `spectrogram.py` — numpy STFT → uint8 log-magnitude grid, base64'd into `/api/synthesize` and `/api/sentence/audio` responses for canvas rendering
- `synth.py` — `synthesise(sentence, phonemes_only=False)` tokenises, walks the resolver chain per token, falls back to per-phoneme via CMUdict. Returns `(wav, skipped, units)` where `units` carries per-word and per-phoneme ms spans for annotation
- `resolver.py` — **the extensibility seam** for top-level tokens. `RESOLVERS = [WordResolver(), PhonemeResolver()]` is walked in order. Add new fallback strategies by appending to this list
- `books.py` — sentence sampling from `data/books/*.txt` (Gutenberg corpus); filters to 30–250 char sentences

### Frontend (vanilla JS)

- `static/app.js` — single module, no framework. Two tabs: Train (record words + phoneme grid + word inventory) and Speak (synth + reference recording + book sample picker)
- `static/recorder-worklet.js` — AudioWorklet that captures Float32 audio from the mic
- WAV encoding is done client-side in `encodeWav()` (manual 44-byte header + int16 PCM). The server validates format but doesn't re-encode
- Phoneme editor (in `app.js`) renders the waveform to canvas with draggable boundary handles; viewport state (`viewStartMs`/`viewEndMs`) handles zoom independently of absolute boundary positions

### Synthesis flow (read `synth.synthesise` if confused)

For each token: try every resolver on the raw token; if all return None and the token isn't itself an ARPABET symbol, phonemise via CMUdict and look up each phoneme with `inventory.best_phoneme_variant(symbol, prev, next)` — this bypasses the resolver chain because the inner slots know their neighbours. Unit selection scores +1 per matching neighbour, tiebreak most-recent. **All-or-nothing per word** — if any phoneme is missing, the whole word is skipped (returned in the `skipped` list, surfaced as red chips in the UI).

## Test conventions

- pytest discovers `tests/test_*.py`
- Inventory tests use `monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)` — never write to the real `inventory/` from tests
- Book tests use `monkeypatch.setattr(books, "BOOKS_DIR", tmp_path)` — same pattern
- `tests/test_app.py::_make_wav_bytes()` is the helper for synthetic 22050 Hz WAV bytes
