# Phoneme extraction from word recordings — design

**Date:** 2026-05-21
**Project:** talky-talky.ai
**Status:** Draft, pending implementation plan
**Builds on:**
- [2026-05-20-word-level-tts-mvp-design.md](2026-05-20-word-level-tts-mvp-design.md)
- [2026-05-20-phoneme-fallback-design.md](2026-05-20-phoneme-fallback-design.md)

## Goal

When the user records a word, automatically propose a slicing of the audio
into per-phoneme spans using the word's CMUdict phoneme sequence, and offer
a waveform editor where the user can refine the boundaries and choose which
slices to save into the phoneme inventory.

The result: one recording session for "hello" can yield WAVs for the word
*and* for HH/AH/L/OW, all derived from the same voice take.

## Locked design decisions

- **Segmentation algorithm:** Duration-weighted slicing. Each ARPABET phoneme
  has a typical-duration prior (vowels ~150 ms, stops ~80 ms, etc.); the audio's
  loud region is allocated proportionally. No external dependencies, no acoustic
  model.
- **Adjustment UI:** Waveform canvas with draggable vertical boundary handles.
  Standard audio-editor interaction. The user *will* need to fix many boundaries
  by hand because pure-duration priors are not acoustic.
- **Trigger:** Inline after every word save (auto-open) AND on a per-chip ✎
  button so the user can re-extract later without re-recording.
- **Overwrite policy:** Per-phoneme toggle in the editor. Defaults: already-recorded
  phonemes have their save-toggle OFF (don't clobber a deliberately-recorded
  isolated phoneme); never-recorded phonemes default ON.
- **Silence handling:**
  - Leading/trailing silence is detected in `propose_boundaries` (same 2%-of-peak
    threshold as `audio.trim_silence`, but uncapped). Boundaries `[0]` and `[N]`
    are set to the first/last loud samples — silent edges are excluded from any
    phoneme slice by default.
  - Internal (word-medial) silence is NOT detected. The user fixes
    word-internal pauses by dragging boundaries across the gap manually.
- **Architecture:** Server holds the audio operations (slicing, peak
  normalisation, WAV writing). Frontend is a pure visual editor sending back
  boundary times in milliseconds.

## Architecture

```
talky-talky.ai/
  app.py                    # + POST /api/extract/propose, POST /api/extract/save
  audio.py                  # + slice_samples(samples, start_ms, end_ms)
  extraction.py             # NEW: PHONEME_DURATION_MS + propose_boundaries
  static/
    app.js                  # + extraction editor (waveform canvas + drag handles)
    style.css               # + editor styles
  templates/index.html      # unchanged — editor is built dynamically in JS
  tests/
    test_extraction.py      # NEW
    test_audio.py           # + slice_samples tests
    test_app.py             # + propose/save route tests
```

No new external dependencies. No format changes — slices are 16-bit mono PCM @
22050 Hz, same as everything else.

## Components

### `extraction.py` (new)

```python
PHONEME_DURATION_MS: dict[str, int]   # ARPABET symbol -> typical duration

def propose_boundaries(
    samples: np.ndarray,
    phoneme_seq: list[str],
    sample_rate: int = SAMPLE_RATE,
) -> list[int]:
    """Return N+1 boundary times in milliseconds for N phonemes.

    First and last boundaries are at the first/last loud sample (peak * 2%),
    NOT at 0 and audio_duration_ms. Silent edges are excluded.

    Internal silences are not detected.
    """
```

**Duration table** covers all 39 ARPABET phonemes, grouped by phonetic class:

| Class | Members | Duration (ms) |
|---|---|---|
| Vowels | AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW | 150 |
| Stops | B D G K P T | 80 |
| Nasals | M N NG | 90 |
| Fricatives | DH F HH S SH TH V Z ZH | 120 |
| Liquids | L R | 90 |
| Glides | W Y | 80 |
| Affricates | CH JH | 130 |

These values are linguistic order-of-magnitude estimates, not measurements
from any particular speaker. The point is *ratios*: a vowel slice should be
roughly twice the width of a stop slice. The user fixes absolute positions
in the editor.

### `audio.py` — `slice_samples` (new helper)

```python
def slice_samples(samples: np.ndarray, start_ms: int, end_ms: int,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return the samples between two millisecond timestamps.

    Out-of-range inputs are clamped to the valid sample range.
    An inverted range (end <= start) returns an empty array.
    """
```

### `POST /api/extract/propose`

Request: `{"word": "<word>"}`.

Server steps:

1. `inventory.normalise(word)`. If empty → 400.
2. `inventory.has(word)`. If false → 404.
3. `phonemes.word_to_phonemes(word)`. If None → 400 (`"word not in dictionary"`).
4. Read WAV: `samples, _ = audio.read_wav(inventory.path(word))`.
5. Compute `audio_duration_ms = len(samples) * 1000 / SAMPLE_RATE`.
6. Compute `boundaries_ms = extraction.propose_boundaries(samples, phoneme_seq)`.
7. Encode the raw WAV bytes as base64.
8. For each phoneme, attach `recorded: bool` from `inventory.has_phoneme` and
   `example` from `phonemes.PHONEME_EXAMPLES`.

Response:

```json
{
  "word": "hello",
  "audio_duration_ms": 712,
  "audio": "<base64 wav>",
  "phonemes": [
    {"symbol": "HH", "example": "he",  "recorded": false},
    {"symbol": "AH", "example": "hut", "recorded": true},
    {"symbol": "L",  "example": "lee", "recorded": false},
    {"symbol": "OW", "example": "oat", "recorded": false}
  ],
  "boundaries_ms": [95, 245, 335, 460, 645]
}
```

Note the first and last boundary values: they bound the *loud* region, not the
full audio. In this example the speech starts ~95 ms in and ends ~67 ms before
the file ends.

### `POST /api/extract/save`

Request:

```json
{
  "word": "hello",
  "boundaries_ms": [95, 220, 335, 460, 645],
  "save": [true, false, true, true]
}
```

Server steps:

1. Normalise word. If empty → 400.
2. `inventory.has(word)`. If false → 404.
3. Validate request shape: `boundaries_ms` and `save` lengths consistent
   (N+1 and N respectively, with N ≥ 1).
4. Validate `boundaries_ms` is monotonically non-decreasing; values within
   `[0, audio_duration_ms + 5]` (5 ms tolerance for rounding).
5. Look up `phoneme_seq = phonemes.word_to_phonemes(word)`. If None → 400 (defensive).
6. Validate `len(phoneme_seq) == N`.
7. Read the WAV samples once.
8. For each index `i` where `save[i] == True`:
   - `slice = audio.slice_samples(samples, boundaries_ms[i], boundaries_ms[i+1])`.
   - If `len(slice) < 30 ms worth of samples` → add to `skipped` with reason
     `"too short"`; do NOT write.
   - Apply `audio.peak_normalise`.
   - Skip `audio.trim_silence` — boundaries are the user's intent.
   - Encode as WAV via stdlib `wave`.
   - `inventory.save_phoneme(phoneme_seq[i], wav_bytes)` — overwrites any
     existing file (per the toggle UX which let the user explicitly choose
     overwrite).
9. Return `{"saved": [...phoneme symbols...], "skipped": [{"symbol": "...", "reason": "..."}, ...]}`.

### Frontend editor

Built dynamically when triggered. Lives in a `<div id="phoneme-editor">`
inserted after the existing record pane.

Structure (rendered):

```
┌── Extracting phonemes from "hello" ───────────────── × ──┐
│                                                          │
│  ▮▮▮▮ │ ▮▮▮▮▮▮▮▮ │ ▮▮▮▮▮ │ ▮▮▮▮▮▮▮▮▮▮▮▮▮ │ ▮▮▮▮         │ ← canvas + drag handles
│  ░░░░ │░░░░░░░░░░│░░░░░░│░░░░░░░░░░░░░░│░░░ ← silent regions dimmed
│                                                          │
│  ☑ HH  he   [▶]   125 ms                                 │
│  ☐ AH  hut  [▶]   115 ms   already recorded              │
│  ☑ L   lee  [▶]   125 ms                                 │
│  ☑ OW  oat  [▶]   185 ms                                 │
│                                                          │
│  [ Save phonemes ]   [ Cancel ]                          │
└──────────────────────────────────────────────────────────┘
```

- **Waveform canvas:** ~120 px tall, full pane width. Downsample the audio
  to one bucket per pixel column; draw min/max as a vertical line per column.
  Regions below the 2%-peak threshold rendered in a dim grey, loud regions in
  the foreground colour.
- **Boundary handles:** N-1 internal vertical lines plus the two endpoints
  (which are also draggable — the user can include or exclude silence by
  dragging the endpoints). Drag updates `boundaries_ms[i]`. Constraint per
  handle: `boundaries_ms[i-1] + 5 <= new <= boundaries_ms[i+1] - 5` (minimum
  10 ms slice width).
- **Per-phoneme row:** checkbox (save toggle), ARPABET symbol (bold mono),
  example word (italic, muted), play button (▶), live-updating duration in ms,
  small "already recorded" tag if `recorded == true`.
- **Default checkbox state:** ON if `recorded == false`, OFF if `recorded == true`.
- **Play button:** decode the base64 WAV via `AudioContext.decodeAudioData`
  (cached after first play), then play a sub-region with `AudioBufferSourceNode`
  using `start(0, startSec, durationSec)`.
- **Save:** POST `/api/extract/save`; on success, dismiss the editor, refresh
  inventory.
- **Cancel:** dismiss without writing.
- **Dismiss (×):** same as Cancel.

The editor is purely additive — the word recording itself was saved in the
prior step; cancelling the editor doesn't affect that.

## Data flow — inline trigger (post-save)

```
user records "hello", clicks Save →
existing /api/record returns 200 →
JS: fetch /api/extract/propose {word:"hello"} →
server returns audio + phonemes + boundaries →
JS: render <phoneme-editor> below record pane →
user drags handles, toggles checkboxes, clicks Save phonemes →
JS: fetch /api/extract/save {word, boundaries_ms, save: [...]} →
server slices, normalises, writes selected phoneme files →
returns {saved, skipped} →
JS: dismiss editor, toast saved/skipped, refresh inventories
```

If `propose` returns 400 (OOV) or 404 (race: word deleted) → toast a friendly
message, don't open the editor. The word recording is still saved.

## Data flow — chip ✎ trigger

```
user clicks ✎ on a word chip →
JS: fetch /api/extract/propose {word} →
(same flow as above)
```

The ✎ icon is a small button rendered inside each word chip in
`#inventory-chips`. Clicking the chip body (i.e. not the ✎) is unchanged
(currently chips have no click behaviour — the design leaves that as is).

## Tokenisation and case handling

- API accepts any case for `word`; server normalises via `inventory.normalise`.
- Phoneme symbols stored uppercase in code and API; filenames lowercase
  (existing convention).
- All boundary times are integer milliseconds; rounding tolerance is ±5 ms
  at the validation seam.

## Error handling

### `extraction.propose_boundaries`

| Condition | Behaviour |
|---|---|
| Empty `phoneme_seq` | Return `[0]` (degenerate; caller's responsibility not to hit this) |
| All samples below silence threshold | Treat the whole clip as the loud region; log WARNING |
| Audio shorter than `len(phoneme_seq) * 30 ms` | Allocate anyway; the save route filters short slices |
| Unknown phoneme symbol in input | Use default duration 100 ms; log WARNING |

### `audio.slice_samples`

| Condition | Behaviour |
|---|---|
| Out-of-range timestamps | Clamp to valid range |
| `end_ms <= start_ms` | Return empty array |

### `POST /api/extract/propose`

| Condition | Response |
|---|---|
| Missing `word` field or empty after normalisation | 400 `{"error": "word is required"}` |
| Word not in inventory | 404 `{"error": "word not recorded", "word": "<normalised>"}` |
| Word not in CMUdict | 400 `{"error": "word not in dictionary", "word": "<normalised>"}` |
| Word file unreadable (corrupted, etc.) | 500, logged with path |

### `POST /api/extract/save`

| Condition | Response |
|---|---|
| Missing required field | 400 |
| Word not in inventory | 404 |
| `len(save) != len(boundaries_ms) - 1` | 400 `{"error": "save length must equal boundaries_ms length minus one"}` |
| `boundaries_ms` not monotonically non-decreasing | 400 `{"error": "boundaries must be non-decreasing"}` |
| Any boundary outside `[0, audio_duration_ms + 5]` | 400 `{"error": "boundary out of range"}` |
| Word OOV in CMUdict (defensive) | 400 |
| Phoneme count mismatch with CMUdict | 400 |
| Slice `i` with `save[i]==true` is < 30 ms | 200 overall; `skipped[i]` with reason `"too short"` |
| Disk write failure | 500, partial writes left on disk (documented) |

### Frontend

- 400 OOV → toast "<word> isn't in the dictionary; can't extract phonemes." Editor doesn't open.
- 404 → toast "<word> not in inventory."
- 400 with field error → toast the error, leave editor open.
- 200 with non-empty `skipped` → editor closes, toast lists what was skipped.
- Network error → toast, leave editor state.
- Drag past neighbour → clamp at `prev + 5 ms` / `next - 5 ms`.

### Logging

- INFO on route entry, with word and boundary count.
- INFO on each saved phoneme.
- WARNING on validation failures and `extraction` edge cases.

## Testing strategy

**Test framework:** `pytest`, run via `uv run pytest`.

### New: `tests/test_extraction.py`

| Test | Verifies |
|---|---|
| `test_propose_boundaries_basic` | 4 phonemes, 800 ms of loud audio → 5 monotonic boundaries spanning the loud region. |
| `test_propose_boundaries_proportional_to_priors` | A vowel + a stop in equal-loud-time audio → vowel slice ≈ 150/230 of total (±5 ms tolerance). |
| `test_propose_boundaries_skips_leading_silence` | 200 ms silence + 600 ms tone → first boundary ≈ 200 ms. |
| `test_propose_boundaries_skips_trailing_silence` | 600 ms tone + 300 ms silence → last boundary ≈ 600 ms. |
| `test_propose_boundaries_all_silent_fallback` | All-zero audio → boundaries span the full audio; WARNING logged. |
| `test_propose_boundaries_unknown_phoneme_default` | Fake symbol gets default duration; allocation completes. |
| `test_propose_boundaries_empty_sequence` | Empty phoneme list → `[0]` (or documented degenerate value). |
| `test_phoneme_duration_table_covers_all_arpabet` | `PHONEME_DURATION_MS.keys() == phonemes.PHONEMES_SET`. |

### Extended: `tests/test_audio.py`

| Test | Verifies |
|---|---|
| `test_slice_samples_basic` | 1-second signal → `slice_samples(s, 100, 200)` returns 2205 samples. |
| `test_slice_samples_clamps_out_of_range` | Out-of-range timestamps → clamped, no error. |
| `test_slice_samples_inverted_range_returns_empty` | `end_ms <= start_ms` → empty array. |

### Extended: `tests/test_app.py`

| Test | Verifies |
|---|---|
| `test_extract_propose_returns_audio_and_boundaries` | Saved word file → response has base64 audio, phonemes with examples + recorded flags, correct boundary count, positive duration. |
| `test_extract_propose_404_when_word_not_recorded` | Absent word → 404. |
| `test_extract_propose_400_when_word_oov` | OOV word → 400. |
| `test_extract_propose_400_when_word_missing_field` | Missing `word` → 400. |
| `test_extract_save_writes_selected_phonemes` | All-true save → all phoneme files written. |
| `test_extract_save_skips_phonemes_with_save_false` | `save[i] = false` → that phoneme not written; others are. |
| `test_extract_save_skips_short_slices` | Slice < 30 ms → returned in `skipped` with `"too short"`. |
| `test_extract_save_rejects_non_monotonic_boundaries` | 400. |
| `test_extract_save_rejects_length_mismatch` | 400. |
| `test_extract_save_overwrites_existing_phoneme` | Pre-existing phoneme + `save[i]=true` → file replaced. |
| `test_extract_save_404_when_word_not_recorded` | 404. |
| `test_extract_save_normalises_word_input` | `word="HELLO!"` succeeds. |

**Not tested**

- Browser editor UI (waveform render, drag interaction, per-phoneme play).
  Manual verification only.
- Subjective audio listenability of slices.
- Snap-to-zero-crossing or other DSP polish.
- Multi-word audio.

**Projected test count:** existing 95 + ~22 new = ~117 total.

## Non-goals (explicit)

- **Forced acoustic alignment.** No HMM, no neural aligner, no MFA.
- **Internal silence detection.** Word-medial pauses are the user's problem
  via the editor.
- **Zero-crossing snap or DSP boundary refinement.**
- **Auto-determining phoneme boundaries from the audio.** Priors only.
- **Multi-pronunciation handling.** Words with multiple CMUdict pronunciations
  use the first (already established in phase 2).
- **OOV phonemisation.** Words not in CMUdict simply can't be extracted; the
  editor doesn't open.
- **Reasons in the API beyond `"too short"`.** Other skip reasons stay
  server-side in logs.
- **Editor preview during recording.** Editor only opens after a successful
  save, not mid-recording.

## Open questions deferred to implementation

- Whether to cache the decoded `AudioBuffer` after the user plays the first
  phoneme, or decode on every play. Default: cache.
- Whether the chip ✎ icon is a separate `<button>` inside the chip, or a
  click-the-symbol-with-a-modifier interaction. Default: separate `<button>`.
- Exact waveform-canvas downsampling — peak-of-bucket vs. RMS-of-bucket.
  Default: peak (visually punchier; matches what speech-editor users expect).
