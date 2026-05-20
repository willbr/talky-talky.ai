# Phoneme fallback (Phase 2) — design

**Date:** 2026-05-20
**Project:** talky-talky.ai
**Status:** Draft, pending implementation plan
**Builds on:** [2026-05-20-word-level-tts-mvp-design.md](2026-05-20-word-level-tts-mvp-design.md)

## Goal

When the existing word-level synthesiser meets a sentence containing a word the
user hasn't recorded, fall back to stitching the word from individually-recorded
phoneme samples. Words that cannot be phonemised (out of dictionary) or whose
phonemes are only partially recorded continue to land in the `skipped` list,
exactly as today.

## Locked design decisions

- **Unit type:** Phonemes (39 ARPABET symbols) recorded in isolation. No
  diphones, no triphones. We accept that isolated-phoneme stitching sounds
  robotic.
- **Grapheme-to-phoneme:** Bundled CMU Pronouncing Dictionary
  (`data/cmudict.txt`). Stress digits stripped at load. Out-of-dict words are
  not phonemised — they go to `skipped` unchanged.
- **Recording UX:** A dedicated third section on the page — a grid of the 39
  ARPABET phonemes. Each phoneme is a chip; clicking arms the existing
  record/stop/save controls. Recorded chips are green, unrecorded are red.
- **Architecture:** Surgical addition. New `phonemes.py`, new `PhonemeResolver`
  added after `WordResolver` in the chain, two new routes. The existing word
  inventory at `inventory/<word>.wav` is unchanged. No breaking changes.
- **All-or-nothing per word:** Within a single word, every phoneme must
  resolve. If any phoneme is missing, the whole word is skipped. No
  half-pronounced words.
- **Intra-word joins:** Phonemes inside one word are concatenated with zero
  silence between them (the existing 5 ms fades handle clicks). Inter-word
  silence stays at 30 ms.

## Architecture

```
talky-talky.ai/
  app.py                            # + 2 routes
  audio.py                          # concatenate() signature widened (see below)
  inventory.py                      # + phoneme_path/has_phoneme/save_phoneme/list_recorded_phonemes
  resolver.py                       # + PhonemeResolver; RESOLVERS updated
  synth.py                          # synthesise() gains phoneme fallback branch
  phonemes.py                       # NEW: ARPABET + CMUdict loader + word_to_phonemes
  data/
    cmudict.txt                     # NEW: bundled CMU Pronouncing Dictionary
  inventory/
    hello.wav                       # words — unchanged
    phonemes/
      aa.wav                        # NEW: per-phoneme recordings
      ae.wav
      ...
  static/
    app.js                          # + phoneme grid + record-phoneme flow
    style.css                       # + grid + chip variants
  templates/
    index.html                      # + <section id="phoneme-pane">
  tests/
    test_phonemes.py                # NEW
    test_resolver.py                # + PhonemeResolver tests
    test_synth.py                   # + phoneme-fallback tests
    test_app.py                     # + 2 routes' tests
    test_audio.py                   # + grouped-concatenate tests
```

**ARPABET set (hardcoded constant in `phonemes.py`):**

```
AA AE AH AO AW AY  B  CH D  DH EH ER EY F  G  HH IH IY JH K
L  M  N  NG OW OY P  R  S  SH T  TH UH UW V  W  Y  Z  ZH
```

39 symbols, matching CMUdict's vocabulary.

**Filesystem mapping:** ARPABET `AA` → `inventory/phonemes/aa.wav`. Filenames are
lowercase to match the existing word inventory convention. Functions accept any
case as input.

**Resolver chain:** `RESOLVERS = [WordResolver(), PhonemeResolver()]`.

- `WordResolver` resolves real English tokens (e.g. `"hello"`).
- `PhonemeResolver` only resolves tokens that ARE ARPABET symbols (e.g. `"aa"`,
  case-insensitive). Returns `None` for any non-phoneme token, even if a file
  by that name exists.
- The synthesis loop is what bridges English-word → phoneme-list. The chain
  itself does not phonemise.

## Components

### `phonemes.py` (new)

```
PHONEMES: tuple[str, ...]                # 39 ARPABET symbols, canonical order
PHONEMES_SET: frozenset[str]             # O(1) membership check
_DICT: dict[str, list[str]]              # loaded from data/cmudict.txt at import

def is_phoneme(token: str) -> bool       # case-insensitive
def word_to_phonemes(word: str) -> list[str] | None   # CMUdict lookup
def list_phonemes() -> list[str]         # the canonical list, for the API
```

`word_to_phonemes("hello")` returns `["HH", "EH", "L", "OW"]`. Returns `None`
for any word not in CMUdict, or empty/whitespace-only input.

CMUdict load happens once at module import. Lines starting with `;;;` are
comments, skipped. Multiple pronunciations (entries like `HELLO(2)`) — keep the
first; ignore the rest. Stress digits stripped (`AH1` → `AH`). Any phoneme
remaining after strip that is not in `PHONEMES_SET` triggers a `WARNING` log
and the entry is skipped.

### `inventory.py` (extended)

Add alongside the existing helpers:

```
PHONEME_INVENTORY_DIR: Path = INVENTORY_DIR / "phonemes"

def phoneme_path(phoneme: str) -> Path                     # case-insensitive input
def has_phoneme(phoneme: str) -> bool
def save_phoneme(phoneme: str, wav_bytes: bytes) -> None   # creates dir if missing
def list_recorded_phonemes() -> list[str]                  # returns uppercase ARPABET symbols
```

Word helpers (`path`, `has`, `save`, `list_words`) are unchanged.

### `resolver.py` (extended)

```python
import phonemes

class PhonemeResolver:
    """Resolves an ARPABET phoneme token to its recorded WAV file.

    Does NOT phonemise words; that is the synthesis loop's job.
    """
    def resolve(self, token: str) -> Path | None:
        if not phonemes.is_phoneme(token):
            return None
        return inventory.phoneme_path(token) if inventory.has_phoneme(token) else None


RESOLVERS: list[Resolver] = [WordResolver(), PhonemeResolver()]
```

### `synth.py` — updated `synthesise`

The synthesis loop becomes a two-pass try per token:

1. **Direct chain pass:** For each token, walk `RESOLVERS`. If any returns a
   `Path`, use it. (Hits the WordResolver for recorded words and the
   PhonemeResolver for tokens that happen to be ARPABET symbols.)

2. **Phonemise pass (fallback):** If the direct pass missed AND the token is
   not itself an ARPABET symbol, call `phonemes.word_to_phonemes(token)`.

   - Returns `None` → add the token to `skipped`.
   - Returns a phoneme list → walk the chain on each phoneme. If every phoneme
     resolves, the word becomes a group of phoneme paths. If any phoneme
     misses, the whole word goes to `skipped` (all-or-nothing).

The output structure changes: instead of building a flat `list[Path]`,
`synthesise` now builds `list[list[Path]]` — outer list = words, inner list =
the paths that make up that word (length 1 for a word file, length N for a
phoneme-stitched word). This is then passed to `audio.concatenate(groups)`.

### `audio.py` — `concatenate` signature widened

`concatenate` accepts two call shapes:

- `concatenate(paths: list[Path]) -> bytes` — flat list. Existing behaviour:
  reads each, applies fades, appends 30 ms silence between adjacent items.
  (No existing call sites of this signature after this change, but kept for
  test simplicity and the principle of not breaking working API.)
- `concatenate(groups: list[list[Path]]) -> bytes` — grouped. For each inner
  group, reads each path, applies fades, appends with **zero** silence inside
  the group. Between groups, the existing 30 ms silence is appended.

Detection: if the first element is a list, treat as grouped. Otherwise flat.

Empty inner groups are silently dropped.

### Routes

#### `POST /api/record/phoneme`

Request: `multipart/form-data` with `phoneme` (string, any case) and `audio`
(WAV blob). Optional query param `overwrite=true`.

Validation pipeline (in order):

1. `phoneme.upper()` must be in `PHONEMES_SET`. Else 400.
2. `audio` file must be present. Else 400.
3. WAV must be 16-bit mono PCM at 22050 Hz. Else 400.
4. After silence trim, length ≥ 100 ms. Else 400.
5. Phoneme already recorded and `overwrite` not set → 409.

On success: peak-normalise the trimmed samples, write
`inventory/phonemes/<lower>.wav`, return 200 with
`{"phoneme": "<UPPER>"}`.

#### `GET /api/phoneme-inventory`

Always returns the canonical 39 phonemes:

```json
{
  "phonemes": [
    {"symbol": "AA", "recorded": true},
    {"symbol": "AE", "recorded": false},
    ...
  ]
}
```

Symbol case is uppercase. Order matches `phonemes.PHONEMES`. Missing
`inventory/phonemes/` directory is treated as "no phonemes recorded yet";
no error.

#### `POST /api/synthesize` (existing, behaviour change only)

The response shape (`{"audio": <base64 or null>, "skipped": [...]}`) is
unchanged. The phoneme fallback is transparent to the API client. A word that
gets synthesised via phonemes appears nowhere special in the response.

## Data flow — recording a phoneme

```
user clicks an unrecorded ARPABET chip (e.g. AA) →
JS sets currentPhoneme="AA", enables record/stop/save, shows "Recording AA..." →
user clicks Record → mic capture → Stop → preview <audio> shows playback →
user clicks Save → JS POST multipart {phoneme: "AA", audio: <wav blob>} to /api/record/phoneme →
server validates symbol, runs trim+length+normalise, writes inventory/phonemes/aa.wav →
JS refetches /api/phoneme-inventory and re-renders chips (AA now green)
```

## Data flow — synthesis with phoneme fallback

```
sentence "the quick brown fox" →
tokenise → ["the", "quick", "brown", "fox"] →

for each token:
  Direct chain pass:
    WordResolver → "the" hits if recorded as a word
    PhonemeResolver → "quick" is not ARPABET → miss
  
  Phonemise pass (if direct missed):
    word_to_phonemes("quick") → ["K", "W", "IH", "K"]
    For each phoneme, walk chain:
      "K" → PhonemeResolver hits → inventory/phonemes/k.wav
      ...
    If all four resolve → group = [k.wav, w.wav, ih.wav, k.wav]
    If any missing → skip "quick"

groups = [[the.wav], [k.wav, w.wav, ih.wav, k.wav], [b.wav, r.wav, aw.wav, n.wav], ...]
audio.concatenate(groups) → final WAV bytes
```

## Tokenisation and case-handling

- Sentence tokenisation is unchanged (`synth.tokenise`).
- CMUdict lookup uppercases the token before lookup.
- Phoneme files are stored lowercase but symbols in code and API are
  uppercase. The `inventory.phoneme_path` / `has_phoneme` / `save_phoneme`
  functions handle case conversion internally.
- The ARPABET set has no stress digits (we collapse `AH0/AH1/AH2` → `AH` at
  CMUdict load).

## Error handling

### `phonemes.py` (import time)

| Condition | Behaviour |
|---|---|
| `data/cmudict.txt` missing | Raise `FileNotFoundError` at import. Fail fast. |
| Malformed CMUdict line | Log `WARNING`, skip the line, continue load. |
| Pronunciation contains non-ARPABET symbol after stress strip | Log `WARNING`, skip that entry. |

### `phonemes.word_to_phonemes`

| Condition | Returns |
|---|---|
| Word in dict | `list[str]` of ARPABET phonemes |
| Word not in dict | `None` |
| Empty / whitespace input | `None` |

### `PhonemeResolver.resolve`

| Condition | Returns |
|---|---|
| Token is not an ARPABET symbol | `None` |
| Token is ARPABET but file not present | `None` |
| Token is ARPABET and file present | `Path` |

### `synth.synthesise` (phoneme fallback path)

| Condition | Behaviour |
|---|---|
| Word in CMUdict, all phonemes recorded | Word is synthesised via phoneme stitch |
| Word not in CMUdict | Word added to `skipped` |
| Word in CMUdict, any phoneme missing | Whole word added to `skipped` (all-or-nothing) |

### `POST /api/record/phoneme`

| Condition | Response |
|---|---|
| Missing `phoneme` field | 400 `{"error": "phoneme is required"}` |
| `phoneme.upper()` not in ARPABET set | 400 `{"error": "unknown phoneme", "allowed": [...39...]}` |
| Missing `audio` field | 400 `{"error": "missing audio field"}` |
| WAV format wrong | 400 `{"error": "expected 16-bit mono PCM at 22050 Hz"}` |
| Too short after trim | 400 `{"error": "recording too short"}` |
| Already recorded, no overwrite | 409 `{"error": "phoneme exists", "phoneme": "AA"}` |
| Disk write failure | 500, logged |

### `GET /api/phoneme-inventory`

Cannot fail in normal operation. Missing directory → returns all 39 with
`recorded: false`.

### `POST /api/synthesize`

No new error codes. Response shape unchanged.

### Frontend

- Phoneme chip click while recording in progress → ignored (matches existing
  word recording behaviour).
- Mic permission denial → same toast/status pattern as the word recorder.
- Network errors on phoneme save → toast, retry button, no inventory refresh.

## Testing strategy

**Test framework:** `pytest`, run via `uv run pytest`.

### New: `tests/test_phonemes.py`

| Test | Verifies |
|---|---|
| `test_arpabet_set_size` | `len(PHONEMES) == 39` and matches `PHONEMES_SET`. Guards against accidental edits. |
| `test_is_phoneme_case_insensitive` | `"AA"`, `"aa"`, `"Aa"` all true; `"hello"` false. |
| `test_word_to_phonemes_known` | `word_to_phonemes("hello") == ["HH", "EH", "L", "OW"]`. |
| `test_word_to_phonemes_case_insensitive` | `"HELLO"`/`"hello"`/`"Hello"` return identical lists. |
| `test_word_to_phonemes_unknown` | Returns `None` for an obviously-OOV word. |
| `test_stress_markers_stripped` | A vowel-heavy word's phonemes have no trailing digits. |
| `test_word_to_phonemes_empty` | Empty / whitespace input returns `None`. |

### Extended: `tests/test_resolver.py`

| Test | Verifies |
|---|---|
| `test_phoneme_resolver_hit` | `resolve("aa")` returns the expected path when file present. |
| `test_phoneme_resolver_miss_no_file` | File absent → `None`. |
| `test_phoneme_resolver_rejects_non_arpabet` | `resolve("hello")` returns `None` even with a `hello.wav` file present (it's the word inventory, not phoneme). |
| `test_phoneme_resolver_case_insensitive` | `"aa"`, `"AA"`, `"Aa"` all hit the same path. |
| `test_resolvers_list_order` | `RESOLVERS` has `WordResolver` first, then `PhonemeResolver`. |

### Extended: `tests/test_synth.py`

| Test | Verifies |
|---|---|
| `test_synthesise_phoneme_fallback_full_coverage` | Word "hello" not recorded but all four phonemes are → audio produced, `skipped == []`. |
| `test_synthesise_phoneme_fallback_partial_missing` | Same setup but one phoneme missing → "hello" in `skipped`. |
| `test_synthesise_phoneme_fallback_word_not_in_cmudict` | OOV word → `skipped`. |
| `test_synthesise_word_resolver_takes_precedence` | Word AND all phonemes recorded → output matches the word file (not the phoneme stitch). |
| `test_synthesise_mixed_word_and_phoneme_words` | Sentence with one recorded word and one phoneme-stitched word — both synthesised, `skipped == []`. |

### Extended: `tests/test_app.py`

| Test | Verifies |
|---|---|
| `test_phoneme_inventory_lists_all_39` | All 39 phonemes returned, unrecorded ones show `recorded: false`. |
| `test_phoneme_inventory_reflects_saved` | Save aa + b → those entries show `recorded: true`. |
| `test_record_phoneme_saves_valid_wav` | `POST /api/record/phoneme` with valid WAV → 200, file written. |
| `test_record_phoneme_rejects_unknown_symbol` | Bad symbol → 400 with `allowed` list. |
| `test_record_phoneme_case_insensitive` | `aa`/`AA` both accepted; stored at `aa.wav`. |
| `test_record_phoneme_409_on_existing` | Conflict without overwrite → 409. |
| `test_record_phoneme_overwrite_true` | With `overwrite=true` → 200 and replaces. |
| `test_record_phoneme_too_short` | <100 ms recording → 400. |
| `test_record_phoneme_rejects_wrong_sample_rate` | Format mismatch → 400. |

### Extended: `tests/test_audio.py`

| Test | Verifies |
|---|---|
| `test_concatenate_grouped_inserts_silence_between_groups_only` | `[[a, b], [c]]` → `a + b + 30ms + c`, no silence between a and b. |
| `test_concatenate_backward_compatible_flat_list` | `[a, b, c]` still inserts 30 ms silence between every adjacent pair. |
| `test_concatenate_grouped_empty_inner_group` | `[[], [a]]` returns same as `[[a]]`. |

### Not tested

- The browser UI for the phoneme grid (manual verification only, deferred to
  acceptance run).
- Subjective audio listenability.
- CMUdict completeness.
- Performance of phoneme-stitched synthesis (small inventories make it
  irrelevant for the MVP).

**Projected test count:** existing 51 + ~25 new = ~76 total.

## Non-goals (explicit)

- **Diphones, triphones, or any contextual units.** Phonemes in isolation
  only.
- **Neural G2P for OOV words.** CMUdict only.
- **Stress preservation.** Stress digits are dropped at load.
- **Multiple pronunciations.** First entry in CMUdict wins; `HELLO(2)` etc.
  are ignored.
- **Carrier-word phoneme segmentation.** Recording UX is one isolated
  utterance per phoneme.
- **Reasons in the API `skipped` list.** Words that fail for different reasons
  (OOV vs missing phoneme) look identical to the client.
- **Migration of the existing word inventory.** Words stay at
  `inventory/<word>.wav`. No moves, no renames.
- **Phoneme inventory completeness gating.** The user can synthesise even with
  zero phonemes recorded — they'll just see more skipped words.

## Open questions deferred to implementation

- Exact CMUdict source URL/version to bundle. The standard is the official
  CMU release; implementation plan will pin a specific version.
- Whether `phonemes.py` lazy-loads the dict on first lookup or eagerly at
  import. Default is eager (simpler, ~100 ms cost at startup).
- Grid layout details (rows vs flex-wrap, chip sizes). Visual tuning during
  the frontend task of the plan.
