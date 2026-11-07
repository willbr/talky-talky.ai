# SQLite unit store + multiple word copies — design

**Date:** 2026-11-07
**Project:** talky-talky.ai
**Status:** Draft, awaiting review
**Builds on:** [2026-05-21-phoneme-variants-design.md](2026-05-21-phoneme-variants-design.md)
**Followed by:** the reference-sentence splitter (separate design, not yet written)

## Why

The synth is currently unintelligible. The user's diagnosis, which this
design accepts: one recording per word and per phoneme isn't enough. The
fix is to keep **many copies** of both words and phonemes and pick
the copy whose recorded context best matches the target.

The next project, the splitter, will cut every reference sentence into
word and phoneme copies, about 90 per sentence. That turns the inventory
from ~125 recordings into thousands, so the storage moves to SQLite
now, before the copies arrive. The user wants the hand-corrected data
to later train neural nets **from scratch, on this project's own data,
using tinygrad**. No pretrained models, ever. This project has no ML in it, but its
storage must make the data easy to query as a dataset later.

## Goals

1. All recording metadata lives in a SQLite database; audio stays as WAV
   files.
2. Words get multiple copies, exactly as phonemes already do. Recording
   an existing word appends a copy; it never overwrites and never asks.
3. Synth picks the best word copy by context: previous word, next word,
   and position in the phrase.
4. Existing data (86 words, 39 phoneme variants, 2 reference sentences) is
   migrated automatically on first start, with nothing lost.
5. The schema is ready for the splitter: sentence-sourced copies and
   alignment rows have a place to go, even though nothing writes them yet.

## Non-goals

- The splitter itself (DTW proposals, alignment editor): next design.
- Any neural-net code or tinygrad integration.
- Changing how phoneme copies are scored (prev/next within word, +1 each,
  newest wins). Whether phoneme context should cross word boundaries is
  a splitter-design question.
- Crossfades, pitch/energy smoothing, or any other concat-quality work.

## Anti-goals (would be a failure even if it works)

- Losing or silently altering any existing recording during migration.
- Storing audio as BLOBs in the database. The user wants WAVs they can
  play, open in Audacity, or delete by hand.
- Adding a third-party dependency for this (no ORM, no migration
  framework). Stdlib `sqlite3` only.
- Tests touching the real `inventory/` or `inventory/talky.db`.

## Constraints

- Audio format unchanged: 16-bit mono PCM, 22050 Hz, validated as today.
- uv only (`uv run …`); no new packages.
- Single-user local Flask app. No concurrency story beyond SQLite's own
  locking; one connection per request is fine.
- Tests redirect storage via `monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)`
  today. That pattern must keep working: the DB path derives from
  `INVENTORY_DIR` at call time, not import time.

## Decisions

### Storage layout

- Database: `inventory/talky.db`, created on demand.
- Audio: `inventory/audio/<id>.wav`, one flat folder. The database stores
  each file's path relative to `INVENTORY_DIR`. Folder layout carries no
  meaning.
- IDs are the existing `%Y%m%dT%H%M%S%f` UTC timestamp format, which sorts
  chronologically. They are **globally unique** across units and
  sentences (one namespace, one flat audio folder). Every endpoint
  addresses a copy by ID alone. Keep `is_valid_variant_id()`-style
  validation for any ID that ends up in a filesystem path.
- *Designer's call:* if a generated ID already exists, bump it by 1 µs
  until it's free.
- Reference sentence audio also moves to `inventory/audio/<id>.wav`; the
  text moves into the database.

### Schema (shape, not final DDL)

**units**: every recorded copy of a word or phoneme.

| column | meaning |
|---|---|
| id | timestamp ID, primary key |
| kind | `word` or `phoneme` |
| label | normalised word, or upper-case ARPABET symbol |
| wav_path | relative to `INVENTORY_DIR` |
| duration_ms | from the WAV |
| source | `isolated` (recorded on its own), `word` (cut from a word copy), or `sentence` (cut from a reference; used by the splitter) |
| source_unit_id | for `source=word`: the word copy it was cut from (null for migrated phonemes whose source copy is unknown) |
| source_sentence_id | for `source=sentence`; null otherwise |
| prev, next | neighbour label: previous/next **word** for word copies, previous/next **phoneme** for phoneme copies; null at edges / when isolated |
| position | word copies only: `mid`, `comma`, `final`, or null (isolated/unknown) |
| source_word | display text for migrated phoneme variants (the old sidecar `source` string), so "from 'hello'" still shows |

Index on `(kind, label)`.

**sentences**: reference recordings. `id`, `text`, `wav_path`,
`created_at`.

**alignment_segments**: created now, empty until the splitter.
`sentence_id`, `tier` (`word`/`phoneme`), `idx`, `label`, `start_ms`,
`end_ms`, `word_idx` (phoneme rows → parent word row). Shaped like Praat
TextGrid tiers so later export is trivial.

**schema_version**: single integer, for future migrations done by hand
in code (no framework).

### Migration (automatic, on first DB open)

Runs once inside a transaction when the DB has no schema yet:

**IDs during migration (designer's call):** every migrated unit gets a
**fresh** ID minted from its WAV file's modification time (UTC), using
the bump rule above on collisions. Old IDs are not kept. In particular,
all 39 existing phoneme copies share the legacy ID
`00000000T000000000000`, so keeping it is impossible. Using mtime keeps
"newest wins" meaningful. Reference sentences keep their existing IDs:
they're already unique timestamps.

1. Each `inventory/<word>.wav` → a `units` row (`kind=word`,
   `source=isolated`, prev/next/position null), file moved to
   `inventory/audio/<new-id>.wav`.
2. Each `inventory/phonemes/<sym>/<id>.{wav,json}` → a `units` row
   carrying over prev, next, duration; `source=isolated` if the
   sidecar said `isolated`, else `source=word` with `source_word` = the
   sidecar string and `source_unit_id` null. Any leftover flat
   `inventory/phonemes/<sym>.wav` is migrated the same way (legacy ID).
3. Each `inventory/sentences/<id>.{wav,txt}` → a `sentences` row.

Safety: copy files first, commit the DB, then delete the originals. A
crash leaves the originals intact and the migration re-runs. Afterwards
the old `phonemes/` and `sentences/` folders are empty and removed.
Migration logs a one-line summary (counts per kind).

### Hand edits to audio files (designer's call)

The database is the index; the WAVs are the audio.
- **A WAV deleted by hand:** reads skip rows whose file is missing and
  log a warning. Synth never picks them, and listings don't show them.
  Rows are **not** auto-deleted: re-adding the file brings the copy back,
  and nothing is lost silently.
- **A WAV edited by hand (e.g. trimmed in Audacity):** supported. Audio
  is always read from the file, so `duration_ms` is a cached
  convenience. Anything that needs timing (synth unit spans) uses the
  real file length.
- **A WAV dropped into `audio/` by hand:** not indexed. Unsupported; record
  through the app.

### Recording words

- `/api/record` always appends a new word copy (`source=isolated`).
  The 409 / `?overwrite=true` path and the frontend's overwrite confirm
  are removed. Phoneme recording already behaves this way; its leftover
  409 handling in `app.js` goes too.
- Format validation, silence trim and peak normalise are unchanged.

### Word copies UI (Train tab)

Word chips get the same ✎ panel phoneme chips already have: one row
per copy showing source/context (e.g. `isolated`, or `from sentence …,
prev=she next=as, comma`), duration, ▶ play, ✎ trim/split, × delete.
Reuse the phoneme panel code; don't fork it.

The word editor (trim + split into phonemes) operates on a **specific
word copy** by ID. Phonemes saved from it get `source=word`,
`source_unit_id` = that copy, prev/next within the word (as today).

Deleting the last copy of a word removes the chip. Deleting a word copy
does **not** delete phoneme copies cut from it. Their audio is
independent.

### Synth: choosing a word copy

The resolver chain must pass context. `Resolver.resolve` gains a
context argument (prev word, next word, position). `WordResolver`
scores each copy:

- +1 if `prev` matches the target's previous word (both null counts)
- +1 if `next` matches
- +1 if `position` matches
- newest ID wins ties

Position of a target token comes from the raw sentence **before**
normalisation strips punctuation: `comma` if the token is followed by
`,` `;` or `:`; `final` if followed by `.` `!` `?` or it's the last
token; else `mid`. Isolated copies have null position, so they never
score the position point.

Phoneme selection is unchanged: `best_phoneme_variant(symbol, prev,
next)` semantics, now as a query.

### Synth response: which copy was used

Each `units` entry in `/api/synthesize` gains `unit_id` (word copy) and
each phoneme sub-span gains its `unit_id`.

Speak tab edit chips (designer's call): build them from `units`, **one
chip per word occurrence that came from a word copy**, not one per
distinct word. The same word can use different copies in one sentence.
Each chip opens the word editor on exactly that `unit_id`. Words built
from phonemes still get no chip, as today. Editing phoneme copies from
the Speak tab is out of scope. The phoneme `unit_id`s are returned for
annotation and future use.

### API shape (builder may rename)

Mirror the existing phoneme variant endpoints, generalised to units:
list copies for a label, delete a copy, propose/trim/split targeting a
copy by ID (defaulting to most recent when omitted, as today). The
phoneme-specific endpoints can become thin aliases or be replaced. The
frontend is the only client.

## Left to the builder

- Exact DDL, endpoint names, and whether `inventory.py` keeps its name
  or splits (e.g. a `store.py` for DB access).
- Connection handling (per-call `sqlite3.connect` is fine).
- How the old phoneme endpoints map onto the new ones.
- A small `scripts/dump_db.py` that prints the database as readable text,
  to make up for losing hand-readable JSON sidecars (the designer's call;
  keep it tiny).
- Test structure, following the existing conventions: a temp
  `INVENTORY_DIR` per test, `_make_wav_bytes()` for audio. Migration
  must have a test that builds the old on-disk layout in a temp dir and
  checks every recording survives with correct metadata.
- Updating CLAUDE.md's architecture section to the new layout.
