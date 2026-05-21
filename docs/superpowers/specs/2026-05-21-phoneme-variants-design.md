# Multi-instance phoneme inventory + context-aware unit selection — design

**Date:** 2026-05-21
**Project:** talky-talky.ai
**Status:** Draft, pending implementation
**Builds on:**
- [2026-05-20-phoneme-fallback-design.md](2026-05-20-phoneme-fallback-design.md)
- [2026-05-21-phoneme-extraction-design.md](2026-05-21-phoneme-extraction-design.md)
- [2026-05-21-word-trim-editor-design.md](2026-05-21-word-trim-editor-design.md)

## Goal

Stop overwriting phoneme recordings. Every time the user saves a
phoneme — whether by recording it standalone via the phoneme grid or
by extracting it from a word — the save appends a new *variant* with
metadata describing the source context. At synthesis time, the
resolver picks the variant whose recorded neighbours best match the
target word's neighbouring phonemes (classic concatenative-TTS unit
selection), with most-recent-wins as the tiebreak.

The motivation: the user has already invested in word recordings and
in editing per-phoneme spans inside them. Each edit currently
clobbers prior takes of the same phoneme. Treating each save as a
new variant turns that work into a growing context-aware library
without changing any of the recording UX upstream.

## Locked design decisions

- **Storage layout:** `inventory/phonemes/<symbol>/<id>.wav` plus a
  sibling `<id>.json` sidecar. `<symbol>` is lowercase ARPABET;
  `<id>` is a UTC timestamp (same `%Y%m%dT%H%M%S%f` format used in
  `inventory.save_sentence`) so lexicographic sort is chronological.
- **Variant metadata:** `{id, symbol, source, prev, next, duration_ms}`
  where `source` is either `"isolated"` or the source word, and
  `prev`/`next` are the neighbour ARPABET symbols within the source
  word (both null for isolated, individually null at word boundaries).
- **Every save is a new variant.** No "canonical" file. Re-recording
  an isolated phoneme appends; re-extracting from a word appends.
  Variants are pruned via an explicit delete UI, not by overwrite.
- **Lazy legacy migration.** The first call to
  `inventory.list_variants(symbol)` moves any pre-existing flat
  `inventory/phonemes/<symbol>.wav` into the new subdir as
  `legacy.wav` with `source="isolated"` and null neighbours. No
  one-shot script.
- **Selection scoring:** for each candidate variant, +1 if its `prev`
  matches the target's `prev` (including both-None), +1 if its `next`
  matches. Highest score wins; tiebreak by most-recent ID.
- **UI:** the phoneme grid's ✎ button toggles an inline variants
  panel below the chip — one row per variant with source/context
  metadata, ▶ play, ✎ trim, × delete.

## Architecture

```
talky-talky.ai/
  inventory.py    # + variants storage, save_variant, list_variants,
                  #   variant_path, delete_variant, best_phoneme_variant,
                  #   lazy legacy migration
  resolver.py     # PhonemeResolver delegates to best_phoneme_variant(prev=None, next=None)
  synth.py        # phoneme-fallback loop passes prev/next from CMUdict
  app.py          # + GET /api/phoneme/variants
                  # + DELETE /api/phoneme/variants
                  # /api/record/phoneme appends variant; returns variant_id
                  # /api/extract/save records prev/next per saved phoneme
                  # /api/phoneme/propose|trim gain optional variant_id arg
  static/app.js   # phoneme chip ✎ → inline variants panel
  static/style.css # styles for the panel rows
  templates/index.html # no change (chip-internal expansion)
  tests/
    test_inventory.py # variant CRUD, scoring, legacy migration
    test_app.py       # variant routes, save metadata correctness
    test_synth.py     # context-aware selection
```

## Inventory module API

```python
def save_variant(
    symbol: str, wav_bytes: bytes,
    *, source: str, prev: str | None, next: str | None,
) -> str: ...
# Writes <id>.wav + <id>.json under inventory/phonemes/<symbol>/. Returns
# the new variant ID. Computes duration_ms from the WAV header.

def list_variants(symbol: str) -> list[dict]: ...
# Returns metadata dicts oldest-first. Runs the lazy legacy migration
# the first time it sees a flat <symbol>.wav alongside the subdir.

def variant_path(symbol: str, variant_id: str) -> Path: ...

def delete_variant(symbol: str, variant_id: str) -> bool: ...
# Removes both .wav and .json. Returns True if it existed, False otherwise.

def best_phoneme_variant(
    symbol: str, prev: str | None = None, next: str | None = None,
) -> Path | None: ...
# Score = (prev_match ? 1 : 0) + (next_match ? 1 : 0). Highest score wins;
# tiebreak by most-recent ID. Returns None when no variant exists.

def has_phoneme(symbol: str) -> bool: ...
# Now means "at least one variant exists." Reuses list_variants.
```

`save_phoneme` is removed — callers move to `save_variant` with explicit
context arguments. Greppable migration.

## Save flow changes

**`/api/record/phoneme`** (isolated):
```python
variant_id = inventory.save_variant(
    phoneme, wav_bytes,
    source="isolated", prev=None, next=None,
)
return jsonify({"phoneme": phoneme, "variant_id": variant_id}), 200
```

**`/api/extract/save`** (from word): the route already knows the word
and the phoneme sequence. For each saved phoneme at index `i`:
```python
prev = phs[i - 1] if i > 0 else None
nxt  = phs[i + 1] if i < len(phs) - 1 else None
variant_id = inventory.save_variant(
    symbol, wav_bytes,
    source=word, prev=prev, next=nxt,
)
```

Response includes the saved variant IDs alongside the symbols so the
frontend can refresh state if needed.

## Synthesis: context-aware lookup

In `synth.synthesise`'s phoneme-fallback loop:

```python
for i, ph in enumerate(phs):
    prev_ph = phs[i - 1] if i > 0 else None
    next_ph = phs[i + 1] if i < len(phs) - 1 else None
    ph_path = inventory.best_phoneme_variant(ph, prev_ph, next_ph)
    if ph_path is None:
        all_resolved = False
        break
    phoneme_paths.append(ph_path)
```

For raw-token resolution (the user typed `HH` directly), `PhonemeResolver`
keeps its single-arg signature and internally calls
`best_phoneme_variant(symbol, None, None)`. The resolver chain in
`resolver.RESOLVERS` is unchanged.

## New endpoints

**`GET /api/phoneme/variants?symbol=HH`** → 200

```json
{
  "symbol": "HH",
  "variants": [
    {"id": "...", "source": "hello", "prev": "AH", "next": "L", "duration_ms": 95},
    {"id": "...", "source": "isolated", "prev": null, "next": null, "duration_ms": 120}
  ]
}
```

400 if symbol is missing or not in `PHONEMES_SET`.

**`DELETE /api/phoneme/variants?symbol=HH&id=<variant-id>`** → 204 on
success, 404 if either the symbol or the ID does not exist.

**`/api/phoneme/propose`** gains optional `variant_id`. Without it, falls
back to the most-recent variant (preserves today's call sites).

**`/api/phoneme/trim`** likewise gains optional `variant_id`. Default
"most recent" again. The crop overwrites the targeted variant in place
without spawning a new variant — that matches the trim semantics for
word recordings.

## Frontend — phoneme chip expansion

Each recorded chip in the phoneme grid grows an inline variants panel
when the user clicks ✎. The chip's outer element stays a flex
container, but switches to `display: block` (or wraps in a vertical
flex) when expanded so the variant rows stack underneath.

Variant row layout:

```
+----------------------------------------------------------------+
|  from "hello"   prev=AH  next=L   95 ms    ▶   ✎   ×            |
|  isolated                        120 ms    ▶   ✎   ×            |
+----------------------------------------------------------------+
```

- ▶ plays via the existing Web Audio playSlice path (fetch + decode +
  play). Each variant has its own URL.
- ✎ opens the existing phoneme trim editor scoped to that variant ID
  (passes `variant_id` through `/api/phoneme/propose` and
  `/api/phoneme/trim`).
- × confirms then DELETEs the variant; on success the row is removed
  and the recorded-count badge in the status strip refreshes.

If a phoneme has zero variants, the chip shows no ✎ (today's
non-recorded state). The recorded class still toggles by
`hasAnyVariants(symbol)`.

## Error handling

| Scenario | Status | Body |
|---|---|---|
| variants?symbol= missing | 400 | `{"error": "symbol is required"}` |
| variants?symbol= unknown | 400 | `{"error": "unknown phoneme"}` |
| DELETE without `id` | 400 | `{"error": "id is required"}` |
| DELETE unknown id | 404 | `{"error": "variant not found"}` |
| trim/propose with unknown variant_id | 404 | `{"error": "variant not found"}` |
| best_phoneme_variant called on a symbol with no variants | returns None (silent — synth.py skips the word) |

## Testing

**`tests/test_inventory.py`** (new functions):

- `test_save_variant_writes_pair`: WAV + JSON both exist; sidecar has
  expected fields.
- `test_list_variants_chronological`: two saves in sequence — listed
  oldest first.
- `test_delete_variant_removes_pair`: both files gone; second delete
  returns False.
- `test_variant_path_returns_wav`: round-trip with save_variant output.
- `test_best_variant_prefers_full_context_match`: variants with
  (AH,L), (AH,None), (None,None) — query (AH,L) returns the first.
- `test_best_variant_prefers_more_recent_on_tie`: two variants with
  identical scoring — the newer ID wins.
- `test_best_variant_returns_none_when_empty`: no variants → None.
- `test_legacy_flat_file_is_migrated_once`: pre-existing
  `inventory/phonemes/hh.wav` becomes `hh/legacy.wav` plus metadata
  on first list_variants; second call sees only the new layout and
  does not re-migrate.

**`tests/test_app.py`**:

- `test_record_phoneme_returns_variant_id`.
- `test_extract_save_records_prev_next`: extract from a 3-phoneme
  word, inspect the saved sidecars — first has prev=null, middle has
  both, last has next=null.
- `test_variants_route_lists_all`.
- `test_variants_route_400_unknown_symbol`.
- `test_variant_delete_204_then_404`.
- `test_phoneme_propose_accepts_variant_id`.
- `test_phoneme_trim_targets_named_variant`: trimming a non-most-recent
  variant overwrites only that one.

**`tests/test_synth.py`**:

- `test_phoneme_selection_uses_context_match`: seed two variants for
  HH — one with (AH,L) context, one isolated. Synthesise a word whose
  HH context is (AH,L). The output should byte-equal the contextual
  variant.
- `test_phoneme_selection_falls_back_to_recent_on_no_match`: only
  isolated variant exists; synth still produces output and uses it.

## Migration notes

- The legacy migration runs on first `list_variants(symbol)` call,
  inside a try/except OSError so test-double inventories don't break.
- Existing `save_phoneme` callers (just two routes in `app.py`) get
  rewritten in this change.
- `phoneme_path(symbol)` is removed. Its only caller is
  `PhonemeResolver`, which is rewritten to call
  `best_phoneme_variant(symbol)` directly (which already returns None
  when no variants exist).
- `has_phoneme(symbol)` semantics widen from "the flat file exists"
  to "any variant exists." No call sites need behavioural changes.

## Out of scope

- DTW-aligned averaging across variants.
- Spectral-similarity rescoring beyond exact prev/next match.
- Pitch / energy normalisation across variants at concat time.
- A dedicated variant-browser UI separate from the chip.
- Diphone units (a separate future spec).
- Forced alignment using sentence references (a separate future spec).
