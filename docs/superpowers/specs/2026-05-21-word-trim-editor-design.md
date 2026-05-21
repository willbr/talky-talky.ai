# Word-recording trim in the existing editor — design

**Date:** 2026-05-21
**Project:** talky-talky.ai
**Status:** Draft, pending implementation plan
**Builds on:**
- [2026-05-21-phoneme-extraction-design.md](2026-05-21-phoneme-extraction-design.md)

## Goal

Let the user crop a recorded word in place from the existing phoneme
extraction editor, so excess head/tail silence inside a recording can be
removed without re-recording. The motivating problem is that audible gaps
between concatenated words in synthesis are partly driven by leading and
trailing silence baked into individual word recordings (the auto-trim at
upload caps at `MAX_TRIM_SECONDS = 0.5` per side, so longer silences
survive).

The cropped recording overwrites `inventory/<word>.wav`. Re-recording
remains the escape hatch if a trim goes wrong.

## Locked design decisions

- **Scope:** word recordings only. Phoneme recordings are short and already
  auto-trimmed at upload; reference sentences are not used in synthesis.
  Both are out of scope.
- **Entry point:** the existing ✎ chip on each saved word — `openExtractEditor`
  in `static/app.js`. No new entry point.
- **Action model:** a separate **Save Trim** button alongside Save Phonemes
  and Cancel. Trim does not save phonemes; Save Phonemes does not crop the
  parent WAV. Each action has one effect.
- **Trim points:** the existing draggable outer handles, `boundariesMs[0]`
  (start) and `boundariesMs[N]` (end). They are already constrained, already
  draggable, already used to anchor phoneme distribution. No new handle code.
- **Visual affordance:** translucent gray overlay over the regions before
  `boundariesMs[0]` and after `boundariesMs[N]` on the canvas. At a glance
  the user sees which audio will be discarded. Inner phoneme boundary lines
  are unchanged.
- **Destructiveness:** trim overwrites `inventory/<word>.wav` in place. No
  backup file. No preview round-trip. A `window.confirm` dialog gates the
  click. This matches the rest of the app — recording with `?overwrite=true`
  already destroys the previous WAV.
- **Post-save:** editor re-fetches `/api/extract/propose` for the same word
  so it reflects the new shorter recording with freshly-proposed phoneme
  boundaries. No client-side patching of state.
- **Server-side post-processing of the crop:** peak-renormalise via
  `audio.peak_normalise`. Skip `audio.trim_silence` — the user just chose
  exactly where the silence should end; we trust them.

## Architecture

```
talky-talky.ai/
  app.py                    # + POST /api/extract/trim
  audio.py                  # unchanged (uses existing slice_samples, peak_normalise)
  static/
    app.js                  # + cropped-region overlay in drawWaveform
                            # + Save Trim button + handler
    style.css               # + .save-trim button style (if needed; reuses .save otherwise)
  tests/
    test_app.py             # + trim route tests
```

No new modules. No new audio primitives. The work is one Flask route plus a
button and a canvas-draw addition.

## Server: `POST /api/extract/trim`

**Request body (JSON):**

```json
{ "word": "fox", "start_ms": 120, "end_ms": 480 }
```

**Success response (200):**

```json
{ "word": "fox", "duration_ms": 360 }
```

**Logic (in `app.py`):**

1. Normalise `word` via `inventory.normalise`. 400 if empty.
2. 404 if `inventory.has(word)` is false.
3. Validate `start_ms` and `end_ms`:
   - Both present and numeric → else 400 `"start_ms and end_ms are required"`.
   - `end_ms > start_ms` → else 400 `"end_ms must be greater than start_ms"`.
   - `start_ms >= 0` and `end_ms <= duration_ms + 5` (matches the 5 ms
     tolerance already used in `/api/extract/save`) → else 400
     `"trim out of range"`.
4. Read the WAV via `audio.read_wav(inventory.path(word))`.
5. Slice via `audio.slice_samples(samples, start_ms, end_ms)`.
6. Reject if `len(slice) < int(SAMPLE_RATE * MIN_LENGTH_SECONDS)` (100 ms,
   the same floor used at record time) → 400 `"trim too short"`.
7. Re-normalise loudness via `audio.peak_normalise(slice)`.
8. Encode as 16-bit mono PCM at 22050 Hz (same recipe as `/api/record`) and
   write via `inventory.save(word, bytes)`, which overwrites.
9. Log and return `{"word": …, "duration_ms": new_duration_ms}`.

## Frontend: editor changes (`static/app.js`)

**Cropped-region overlay** — extends `drawWaveform`:

After the existing sample-render loop and before the boundary lines, fill
two rectangles with `rgba(0, 0, 0, 0.55)`:

- left rect: from `x = 0` to `msToPx(boundariesMs[0])`, full canvas height.
- right rect: from `msToPx(boundariesMs[N])` to `x = canvas.width`, full
  canvas height.

If `boundariesMs[0] <= viewStartMs` the left rect renders zero width and is
elided. Same for the right side.

**Save Trim button** — added to `actions` in `buildEditorDom`, between
`saveBtn` (Save phonemes) and `cancelBtn`:

```js
const trimBtn = document.createElement('button');
trimBtn.textContent = 'Save trim';
trimBtn.className = 'save-trim';
trimBtn.addEventListener('click', () => saveTrimFromEditor());
```

Disabled (`trimBtn.disabled = true`) at build time and after every
`drawWaveform` if `boundariesMs[0] === 0 && boundariesMs[N] === audioDurationMs`
— nothing to trim.

**Handler `saveTrimFromEditor`:**

1. `const word = editor.word;`
2. `const start = editor.boundariesMs[0];`
3. `const end = editor.boundariesMs[editor.boundariesMs.length - 1];`
4. `window.confirm("Trim '" + word + "' to " + start + "–" + end + " ms? This overwrites the recording.")` — bail if cancelled.
5. POST to `/api/extract/trim` with `{ word, start_ms: start, end_ms: end }`.
6. On 200: `setStatus("Trimmed " + word + " → " + duration_ms + " ms")`;
   `closeExtractEditor()`; `await openExtractEditor(word)` to reload the
   trimmed audio with fresh boundary proposals.
7. On error: `setStatus("Trim: " + (err.error || r.statusText), true)`;
   editor stays open.

## Data flow

```
User drags boundary[0] / boundary[N]      (existing code)
        │
        ▼
drawWaveform renders shaded overlays      (new)
        │
        ▼
User clicks "Save trim"
        │
        ▼
window.confirm                            (new)
        │
        ▼
POST /api/extract/trim {word, start_ms, end_ms}
        │
        ▼
server slices, renormalises, overwrites   (new route)
        │
        ▼
closeExtractEditor() → openExtractEditor(word)
        │
        ▼
GET /api/extract/propose returns the      (existing route)
shorter audio + re-proposed boundaries
```

## Error handling

| Condition | Status | Body |
|---|---|---|
| empty `word` after normalise | 400 | `{"error": "word is required"}` |
| `word` not in inventory | 404 | `{"error": "word not recorded", "word": …}` |
| `start_ms`/`end_ms` missing or non-numeric | 400 | `{"error": "start_ms and end_ms are required"}` |
| `end_ms <= start_ms` | 400 | `{"error": "end_ms must be greater than start_ms"}` |
| `start_ms < 0` or `end_ms > duration_ms + 5` | 400 | `{"error": "trim out of range"}` |
| slice shorter than 100 ms | 400 | `{"error": "trim too short"}` |

All errors are surfaced to the user via the existing `setStatus(msg, true)`
pattern. No new error UI.

## Testing

**New tests in `tests/test_app.py`** (using the existing `_make_wav_bytes()`
helper and `monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)`
pattern from CLAUDE.md):

- `test_trim_crops_recording`: write a 1 s WAV with a known loud burst at
  500 ms; POST trim 300–700 ms; assert the saved file is ~400 ms long and
  the loud sample survives somewhere in the middle.
- `test_trim_404_unknown_word`: POST trim for a word not in inventory.
- `test_trim_400_inverted_range`: `end_ms <= start_ms` returns 400.
- `test_trim_400_too_short`: 50 ms span returns 400 with `"trim too short"`.
- `test_trim_400_out_of_range`: `end_ms` beyond `duration_ms + 5` returns 400.
- `test_trim_renormalises`: write a quiet WAV (peak much smaller than
  `PEAK_TARGET`); trim a span containing the peak; assert the result is
  peak-normalised to `audio.PEAK_TARGET`.

No frontend tests — the project has no JS test harness today; the
behaviour is exercised via manual click-through.

## Out of scope

- Trimming phoneme recordings.
- Trimming reference-sentence recordings.
- A backup-and-revert workflow.
- A preview-before-commit workflow.
- Mid-recording cuts (removing a noise burst inside the loud region).
- Reshaping fades or applying any other DSP beyond peak normalisation.
- Any change to the synthesis-time inter-word gap (already addressed
  separately by setting `INTER_WORD_SILENCE_SECONDS = 0.0`).
