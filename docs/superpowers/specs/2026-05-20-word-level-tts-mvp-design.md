# Word-level concatenative TTS MVP — design

**Date:** 2026-05-20
**Project:** talky-talky.ai
**Status:** Draft, pending implementation plan

## Goal

Ship a working end-to-end concatenative TTS demo in which a user can:

1. Record audio for individual words in their browser.
2. Type a sentence and hear it played back as a stitched concatenation of the recorded words.

The MVP synthesises *only* words that have been recorded. Words not in the inventory are skipped with a visible warning in the UI, not synthesised by any fallback mechanism. A future phase will add a phoneme-level fallback for unknown words; the architecture must leave a clean seam for that, but no phoneme code is written in the MVP.

## Locked design decisions

- **Phoneme scope:** Word-level only in the MVP. Architecture exposes a resolver chain so a `PhonemeResolver` can be added later without touching the synthesis loop.
- **Vocabulary:** Freeform — the user types a word, records it, saves it. No curated word list, no walkthrough.
- **Unknown words during synthesis:** Skip and warn. Synthesis returns audio of the resolved words plus a list of skipped tokens. UI displays the skipped list as clickable chips that pre-fill the recorder.
- **Architecture:** Single Flask app, single HTML page, two panes (record + synthesise) side-by-side. No DB; filesystem is the inventory.
- **Audio toolchain:** No `ffmpeg`, no `pydub`. Browser captures raw PCM via `AudioContext` and encodes WAV in JavaScript. Server uses stdlib `wave` plus `numpy` for I/O and DSP.

## Architecture

```
talky-talky.ai/
  app.py                  # Flask routes
  inventory.py            # word normalisation, save/has/list
  resolver.py             # WordResolver — seam for future phoneme fallback
  synth.py                # synthesise(sentence) -> (wav_bytes, skipped)
  audio.py                # WAV I/O, trim, fade, concatenate (stdlib wave + numpy)
  inventory/              # <word>.wav files, one per recorded word
  static/
    app.js                # Web Audio capture + WAV encode + UI
    style.css
  templates/
    index.html            # single page, two panes
  tests/
    test_inventory.py
    test_audio.py
    test_resolver.py
    test_synth.py
    test_app.py
    fixtures/
```

**Backend stack:** Flask, numpy. Stdlib `wave` for WAV I/O.
**Frontend stack:** Vanilla JS, Web Audio API (`AudioContext` + `AudioWorkletNode` or `ScriptProcessorNode` fallback). No framework, no build step.

**Resolver chain — the phoneme-future seam:** `synth.synthesise` does not look up word files directly. It calls a list of resolvers in order; each returns a `Path | None`. The MVP ships one resolver, `WordResolver`, which checks `inventory/<normalised>.wav`. When phonemes arrive, `PhonemeResolver` is appended after `WordResolver` in the chain. The synthesis loop never changes.

## Components

| Module | Responsibility |
|---|---|
| `app.py` | Flask routes only. Thin — delegates to the modules below. |
| `inventory.py` | `normalise(word)`, `save(word, wav_bytes)`, `has(word)`, `list_words()`, `path(word)`. Word normalisation is lowercase + strip ASCII punctuation. |
| `resolver.py` | `WordResolver.resolve(word) -> Path \| None`. List `RESOLVERS = [WordResolver()]` is the chain. |
| `synth.py` | `synthesise(sentence) -> (wav_bytes \| None, skipped: list[str])`. Tokenises, runs the resolver chain, concatenates, returns. |
| `audio.py` | `read_wav`, `write_wav`, `trim_silence`, `apply_fade`, `concatenate`. Uses stdlib `wave` + numpy. |

## Routes

- `GET /` — serves `index.html`.
- `GET /api/inventory` — JSON `{"words": [...]}` sorted alphabetically.
- `POST /api/record` — multipart form `{word, audio}`. Validates WAV format, runs the per-recording pipeline, saves. Query param `overwrite=true` to replace an existing word.
- `POST /api/synthesize` — JSON `{"sentence": "..."}`. Returns `{"audio": <base64 wav> | null, "skipped": [...]}`.

## Audio format (enforced everywhere)

- Sample rate: **22050 Hz**.
- Channels: **mono**.
- Sample width: **16-bit signed PCM**.

The browser encoder writes at exactly this format. Browser MUST downmix to mono and resample from the mic's native rate (typically 48 kHz) via `OfflineAudioContext`. The server validates the WAV header on upload and rejects mismatches with a 400. Failing at upload is cheaper than discovering format drift mid-synthesis.

## Data flow — recording

```
user types "hello" → clicks Record →
AudioContext captures mic at native rate (e.g. 48 kHz) →
clicks Stop → OfflineAudioContext resamples to 22050 Hz mono →
JS encodes WAV (44-byte header + int16 samples) →
fetch POST /api/record (multipart: word + audio blob) →
server: normalise word, validate WAV header,
        load samples to np.ndarray[int16],
        trim leading/trailing silence (capped at 500 ms),
        peak-normalise to -3 dBFS,
        write inventory/hello.wav →
return 200 → JS refetches /api/inventory and re-renders the chip list
```

## Data flow — synthesis

```
user types "Hello, world!" → clicks Synthesise →
fetch POST /api/synthesize →
server tokenises → ["hello", "world"] (lowercase, punctuation stripped) →
for each token, resolver chain returns path-or-None →
audio.concatenate([resolved_paths]):
    for each path:
        read samples,
        apply 5 ms linear fade-in and fade-out,
        append to output,
        append 30 ms of silence (skipped after final clip) →
return {audio: base64(wav_bytes), skipped: ["..."]} →
JS plays audio via <audio src="data:audio/wav;base64,..."> and
displays the skipped list as chips that prefill the recorder
```

## Tokenisation rules

- Lowercase.
- Strip ASCII punctuation: `.,!?;:'"()[]—–-`.
- Split on whitespace.
- Each token is looked up independently. No multi-word units, no contractions handling: `"don't"` becomes token `dont`, which the user must record under that filename or accept as skipped.

## Per-recording pipeline (server-side)

1. Parse WAV header; reject if not 16-bit mono PCM at 22050 Hz.
2. Load samples as `np.ndarray[int16]`.
3. Trim leading/trailing silence: find first/last sample with `|s| > peak * 0.02`. Cap trim at 500 ms per side so a quiet inhale isn't deleted.
4. Peak-normalise to -3 dBFS (max abs sample → 23170).
5. Write to `inventory/<word>.wav` via stdlib `wave`.

## Per-synthesis pipeline (server-side)

1. Tokenise sentence.
2. Walk the resolver chain per token; collect `(word, path | None)` pairs.
3. For each resolved path: `read_wav` → `apply_fade(5 ms)` → append.
4. Insert 30 ms of silence between adjacent resolved words.
5. `np.concatenate`, write to `BytesIO`, return bytes + skipped list.

## Error handling

### Recording route

| Condition | Response |
|---|---|
| Empty word after normalisation | 400 `{"error": "word is empty after normalisation"}` |
| Wrong audio format | 400 `{"error": "expected 16-bit mono PCM at 22050 Hz"}` |
| Word exists, no `overwrite=true` | 409 `{"error": "word exists"}` |
| Recording < 100 ms after silence trim | 400 `{"error": "recording too short"}` |
| Disk write failure | 500, logged |

### Synthesis route

| Condition | Response |
|---|---|
| Empty sentence | 400 |
| All tokens missing | 200 with `{"audio": null, "skipped": [...]}` |
| Some tokens missing | 200 with stitched audio + skipped list |
| Corrupted inventory file | 500, logged with filename |

### Frontend

- Mic permission denied → record pane shows permission-error state with retry button.
- Network error on submit → toast + retry button, no inventory refresh.
- `<audio>` playback failure → fallback download link for the returned WAV.

### Logging

Stdlib `logging` at INFO for route entry/exit, WARNING for 4xx, ERROR for 5xx. No structured logging in the MVP.

## Testing strategy

**Framework:** `pytest`. Single dev dependency.

| Module | Tests |
|---|---|
| `inventory.py` | Normalisation cases (`"Hello!"` → `"hello"`, `"..."` → empty); `save`/`has`/`list_words` round-trips. |
| `audio.py` | `read_wav`/`write_wav` round-trip; `trim_silence` on padded and all-silent inputs; `apply_fade` produces monotonic ramps; `concatenate` length and format checks. |
| `resolver.py` | One hit, one miss. |
| `synth.py` | `synthesise("hello world")` with both recorded; mixed-coverage with skipped list; all-missing returns audio=None; case/punctuation-insensitive matching. |
| `app.py` | Flask test client: inventory listing, record success, format-mismatch 400, existing-word 409, synthesise JSON shape. |

**Test fixtures:** ~200 ms real WAV clips committed for two or three words. Synthetic int16 arrays generated via numpy in tests where a specific signal shape is needed.

**Out of MVP scope**

- Browser UI tests (no Playwright). JS is small enough to verify by hand.
- Audio *quality* assertions. Tests check structure, not subjective listenability.
- MediaRecorder mocking.
- CI. Add after the first working demo.

## Non-goals (explicit)

- **Phoneme synthesis.** Architectural seam only; no implementation.
- **Multi-speaker support.** One inventory, one voice.
- **Pitch/duration manipulation.** Words are stitched verbatim.
- **Authentication, user accounts, deployment.** Local dev server only.
- **Mobile-friendly UI.** Desktop browsers only.
- **Audio upload from file.** Mic only.
- **Search/filter on inventory.** Flat alphabetised chip list is enough at MVP scale.

## Open questions deferred to implementation

- Exact JS approach for resampling 48 kHz → 22050 Hz: `OfflineAudioContext` is the obvious tool, but the exact code path is a writing-plans concern.
- Whether `AudioWorkletNode` or `ScriptProcessorNode` is used for capture — `AudioWorklet` is the modern choice but adds the small complexity of a separate worklet file. Pick one in the implementation plan.
- The exact silence-threshold constant and inter-word silence duration are tunables that the implementation plan will treat as named constants for easy adjustment.
