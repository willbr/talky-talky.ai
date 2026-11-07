# SQLite Unit Store + Multiple Word Copies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all recording metadata into `inventory/talky.db`, keep audio as WAVs in `inventory/audio/`, give words multiple copies chosen by context, and migrate existing data automatically.

**Architecture:** A new `store.py` owns the database and the audio folder and replaces the file-layout functions in `inventory.py`, which shrinks to `INVENTORY_DIR` + `normalise`. A new `migrate.py` converts the legacy on-disk layout the first time the database is created. Resolvers gain a `Context` (prev/next word, position) and return unit dicts, so synth can report which copy it used. The frontend generalises the phoneme "variants" panel into a units panel used by both word and phoneme chips.

**Tech Stack:** Python 3.13, Flask 3, numpy, stdlib `sqlite3`, vanilla JS. uv for everything.

**Spec:** `docs/superpowers/specs/2026-11-07-sqlite-unit-store-design.md`

## Global Constraints

- Audio is 16-bit mono PCM at 22050 Hz (`audio.SAMPLE_RATE`, `CHANNELS`, `SAMPLE_WIDTH`); uploads that don't match → 400 as today.
- Stdlib `sqlite3` only. No new dependencies. Run everything via `uv run …`.
- Audio is never stored in the DB. Each unit/sentence is `inventory/audio/<id>.wav`; the DB stores `wav_path` relative to `INVENTORY_DIR`.
- IDs use the `%Y%m%dT%H%M%S%f` UTC format, are unique across units and sentences, and are validated with `inventory.is_valid_id()` (regex `\d{8}T\d{12}`) before touching a path. On collision, bump by 1 µs.
- The DB path is derived from `inventory.INVENTORY_DIR` **at call time**, so `monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)` isolates tests. Tests never touch the real `inventory/`.
- Rows whose WAV is missing are skipped by every read (logged once per id per process), never auto-deleted.
- Word copy scoring: +1 prev word match, +1 next word match, +1 position match (`mid`/`comma`/`final`), both-null counts as a match, newest ID wins ties. Phoneme scoring: prev/next only, unchanged.
- Commit directly to master.

## Review Focus

1. **The real inventory's 39 phoneme copies all share ID `00000000T000000000000`.** All must survive migration with distinct fresh IDs (Task 2: `test_migrate_legacy_phonemes_sharing_an_id_all_survive`).
2. **Punctuation detached from or wrapped around words** (`wait , as`, `"Hello," she said.`). Position must still be computed correctly (Task 3: `test_positions_handle_detached_and_quoted_punctuation`).
3. **A WAV deleted by hand.** Synth falls back to another copy or skips the word, with no 500 (Task 1: `test_missing_file_rows_are_skipped`; Task 3: `test_synth_skips_copy_whose_file_is_missing`).
4. **A unit id from the URL/body that is malformed, a traversal attempt, or belongs to another kind/label.** These return 404, and nothing outside `audio/` is touched (Task 4: `test_unit_routes_reject_bad_ids`, `test_phoneme_trim_rejects_word_unit_id`).
5. **Deleting the last copy of a word** removes it from `/api/inventory`, and phoneme copies cut from it survive with their `source_word` (Task 4: `test_delete_last_word_copy_keeps_cut_phonemes`).

---

### Task 1: `store.py`: database, units, sentences

**Files:**
- Create: `store.py`
- Modify: `inventory.py`: rename `is_valid_variant_id` → `is_valid_id` (keep regex)
- Test: `tests/test_store.py` (new)

**Interfaces:**
- Consumes: `inventory.INVENTORY_DIR`, `inventory.normalise`, `inventory.is_valid_id`.
- Produces (all in `store.py`):
  - `DB_FILENAME = "talky.db"`, `AUDIO_SUBDIR = "audio"`, `POSITIONS = ("mid", "comma", "final")`
  - `db_path() -> Path`, `audio_dir() -> Path`
  - `connect() -> sqlite3.Connection`: creates the DB + schema if absent (Task 2 hooks migration in here); `row_factory = sqlite3.Row`
  - `new_id(reserved: set[str] | None = None, when: datetime | None = None, check_files: bool = True) -> str`
  - `add_unit(kind: str, label: str, wav_bytes: bytes, *, source: str, prev: str | None = None, next: str | None = None, position: str | None = None, source_unit_id: str | None = None, source_sentence_id: str | None = None, source_word: str | None = None) -> str`
  - `get_unit(unit_id: str) -> dict | None`: None for invalid id, no row, or missing file
  - `unit_path(unit_id: str) -> Path`
  - `list_units(kind: str, label: str) -> list[dict]`: oldest first
  - `list_labels(kind: str) -> list[str]`: sorted; labels with ≥1 present copy
  - `delete_unit(unit_id: str) -> bool`
  - `replace_unit_audio(unit_id: str, wav_bytes: bytes) -> int`: overwrites the WAV, refreshes `duration_ms`, returns it; raises `KeyError` if not found
  - `best_unit(kind: str, label: str, *, prev: str | None = None, next: str | None = None, position: str | None = None) -> dict | None`
  - `save_sentence(text: str, wav_bytes: bytes) -> str`, `find_sentence_by_text(text: str) -> Path | None`, `list_sentences() -> list[dict]` (`[{id, text}]`, oldest first)
  - Unit dict keys: `id, kind, label, source, source_unit_id, source_sentence_id, source_word, prev, next, position, duration_ms`
  - Labels are canonicalised inside `store`: words via `inventory.normalise`, phonemes upper-cased.

Schema (verbatim; `schema_version` holds `1`):

```sql
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS units (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('word','phoneme')),
  label TEXT NOT NULL,
  wav_path TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('isolated','word','sentence')),
  source_unit_id TEXT,
  source_sentence_id TEXT,
  source_word TEXT,
  prev TEXT,
  next TEXT,
  position TEXT CHECK (position IN ('mid','comma','final'))
);
CREATE INDEX IF NOT EXISTS units_kind_label ON units(kind, label);
CREATE TABLE IF NOT EXISTS sentences (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, wav_path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alignment_segments (
  sentence_id TEXT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
  tier TEXT NOT NULL CHECK (tier IN ('word','phoneme')),
  idx INTEGER NOT NULL,
  label TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  word_idx INTEGER,
  PRIMARY KEY (sentence_id, tier, idx)
);
```

- [ ] **Step 1: Write failing tests in `tests/test_store.py`** (fixture `temp_inventory` monkeypatches `inventory.INVENTORY_DIR`; WAV helper as in `test_inventory._wav_bytes`):
  - `test_add_unit_writes_wav_and_row`: `uid = store.add_unit("word", "Hello!", wav, source="isolated")`; `store.unit_path(uid) == tmp/"audio"/f"{uid}.wav"` exists; `store.get_unit(uid)["label"] == "hello"`; `duration_ms == 200` for a 0.2 s WAV.
  - `test_phoneme_label_is_uppercased`: `add_unit("phoneme", "aa", …)` → label `"AA"`.
  - `test_list_units_oldest_first` and `test_list_labels_sorted_distinct` (two `hello` copies + one `apple` → `["apple", "hello"]`).
  - `test_new_id_bumps_on_collision`: `when` fixed, `reserved={id0}` → returns id0 + 1 µs.
  - `test_delete_unit`: True then False; WAV gone; invalid id (`"../../x"`) → False.
  - `test_replace_unit_audio_refreshes_duration`.
  - `test_best_word_unit_scoring`: copies (prev=None,next=None,pos=None), (prev="she", next="as", pos="comma"), (prev="she", next=None, pos="mid"); query `prev="she", next="as", position="comma"` → the second; query with no context → the first (scores 3: null prev, null next, and null position all match).
  - `test_best_phoneme_unit_ignores_position`: two phoneme copies identical context, newest wins regardless of `position`.
  - `test_missing_file_rows_are_skipped`: delete a copy's WAV by hand → excluded from `list_units`, `list_labels`, `best_unit`, `get_unit`; row still in DB (query `connect()` directly).
  - `test_sentence_roundtrip`: `save_sentence` writes `audio/<id>.wav`; `find_sentence_by_text("hello,  WORLD")` matches `"Hello world"`; `list_sentences()` returns `[{id, text}]`.

- [ ] **Step 2: Run** `uv run pytest tests/test_store.py -v`. Expected: FAIL (`ModuleNotFoundError: store`).

- [ ] **Step 3: Implement `store.py`.** One short-lived connection per public call (`with closing(connect()) as conn, conn:`). `best_unit` scores in Python over `list_units` (no SQL scoring). Missing-file warnings go through `logging.getLogger("store")`, once per id via a module-level set.

- [ ] **Step 4: Run** `uv run pytest tests/test_store.py -v` → PASS; `uv run pytest` → all existing tests still pass (old `inventory` functions are untouched in this task).

- [ ] **Step 5: Commit** `store.py`, `inventory.py`, `tests/test_store.py`: "Store: SQLite-backed units and sentences".

---

### Task 2: `migrate.py`: legacy layout → database

**Files:**
- Create: `migrate.py`
- Modify: `store.py` (`connect()` creation path)
- Test: `tests/test_migrate.py` (new)

**Interfaces:**
- Consumes: `store.new_id`, schema creation from Task 1.
- Produces: `migrate.migrate(conn: sqlite3.Connection, inventory_dir: Path) -> tuple[dict[str, int], list[Path]]`, which returns counts `{"words", "phonemes", "sentences"}` and the legacy files to delete afterwards. It never deletes anything itself.

Behaviour:
- Legacy sources:
  - `inventory_dir/*.wav` → word units, `source="isolated"`, no context.
  - `phonemes/*/*.wav` + optional sidecar `.json`, and flat `phonemes/*.wav`, become phoneme units. If the sidecar `source` is `"isolated"` or the sidecar is missing/bad, use `source="isolated"`. Otherwise use `source="word"`, `source_word=<sidecar source>`, `source_unit_id=None`. prev/next come from the sidecar. `duration_ms` is recomputed from the WAV.
  - `sentences/<id>.wav` + `.txt` → sentence rows keeping `<id>`. A WAV with no `.txt` is left in place, with a warning.
- Order: sentences first (their IDs go into `reserved`). Then words sorted by label, then phonemes sorted by (symbol, old id). Unit IDs come from `new_id(reserved, when=<file mtime UTC>, check_files=False)`, so re-runs mint the same IDs.
- Files are **copied** to `audio/<id>.wav`, overwriting any copy left by an earlier attempt that crashed.
- `store.connect()` when `talky.db` is absent:
  1. Remove any stale `talky.db.tmp`.
  2. Build the schema in the tmp file, run `migrate`, commit and close.
  3. `tmp.replace(db)`.
  4. Delete the returned originals, then remove the now-empty `phonemes/<sym>/`, `phonemes/` and `sentences/` dirs.
  5. Log `migrate: words=N phonemes=N sentences=N`.

  If the DB exists but legacy files remain (a crash after the rename), log a warning and leave them.

- [ ] **Step 1: Write failing tests in `tests/test_migrate.py`.** Each builds the old layout under `tmp_path`, then calls `store.connect()` (or any store read):
  - `test_migrate_words`: `hello.wav`, `apple.wav` → `store.list_labels("word") == ["apple", "hello"]`; originals gone; `audio/` has 2 WAVs.
  - `test_migrate_legacy_phonemes_sharing_an_id_all_survive`: `phonemes/aa/00000000T000000000000.{wav,json}`, same for `hh` and `s` (same mtime via `os.utime`) → 3 phoneme units with 3 distinct valid IDs; `phonemes/` dir removed.
  - `test_migrate_phoneme_sidecar_context`: sidecar `{"source": "hello", "prev": "HH", "next": "L"}` → unit `source=="word"`, `source_word=="hello"`, `prev=="HH"`, `next=="L"`.
  - `test_migrate_flat_phoneme_file`: `phonemes/aa.wav` → one isolated `AA` unit.
  - `test_migrate_sentences_keep_ids`: `sentences/20260521T194418044438.{wav,txt}` → `list_sentences()[0]["id"]` equals it; `find_sentence_by_text` works; `sentences/` removed.
  - `test_migrate_runs_once`: second `connect()` after migration changes nothing (unit count stable); a `hello.wav` written afterwards is **not** imported.
  - `test_migrate_recovers_from_stale_tmp`: write garbage `talky.db.tmp` first → migration still succeeds.
  - `test_migrate_is_deterministic`: run `migrate.migrate` twice on fresh in-memory connections over the same dir → identical ID lists.

- [ ] **Step 2: Run** `uv run pytest tests/test_migrate.py -v` → FAIL.
- [ ] **Step 3: Implement `migrate.py` and the `store.connect()` creation path.**
- [ ] **Step 4: Run** `uv run pytest` → all pass.
- [ ] **Step 5: Commit**: "Migrate legacy inventory layout into SQLite on first open".

---

### Task 3: Context-aware resolvers and synth on the store

**Files:**
- Modify: `resolver.py`, `synth.py`
- Test: `tests/test_resolver.py`, `tests/test_synth.py` (seed via `store.add_unit`, not by writing files)

**Interfaces:**
- Consumes: `store.best_unit`, `store.unit_path`.
- Produces:
  - `resolver.Context` frozen dataclass `(prev: str | None = None, next: str | None = None, position: str | None = None)`
  - `Resolver.resolve(self, token: str, ctx: Context) -> dict | None` (unit dict). `WordResolver` → `store.best_unit("word", token, prev=, next=, position=)`. `PhonemeResolver` → ARPABET tokens only, `store.best_unit("phoneme", token)` with no context.
  - `synth.tokenise_with_positions(sentence: str) -> list[tuple[str, str]]` (normalised token, position). `synth.tokenise` stays.
  - `synth.synthesise(sentence, phonemes_only=False)` keeps its return shape. Each unit gains `"unit_id"`: the word copy id, or `None` when the word was built from phonemes. Each phoneme span gains `"unit_id"`.

Position rule: look at the punctuation **after** a token, up to the next kept token. That covers trailing punctuation on the raw token and any punctuation-only raw tokens that follow it.
- Any of `. ! ?` → `final`.
- Otherwise any of `, ; :` → `comma`.
- The last kept token is always `final`.
- Otherwise `mid`.

Neighbour words for `Context` are the adjacent **kept tokens in the sentence**, whether or not those neighbours were skipped.

- [ ] **Step 1: Rewrite/extend tests.**
  - `test_resolver.py`: existing cases ported to `store`, plus `test_word_resolver_uses_context`. Two `hello` copies, one with `next="world", position="mid"` (older) and one isolated (newer). `resolve("hello", Context(next="world", position="mid"))["id"]` is the older one, and `resolve("hello", Context())["id"]` is the newer.
  - `test_synth.py`: port seeding to `store.add_unit`. Add:
    - `test_positions_basic`: `"Alice thought, she wait."` → `[("alice","mid"),("thought","comma"),("she","mid"),("wait","final")]`.
    - `test_positions_handle_detached_and_quoted_punctuation`: `'"Hello," she said.'` → hello `comma`, she `mid`, said `final`; `"wait , as"` → wait `comma`, as `final`.
    - `test_synth_picks_word_copy_by_context`: output bytes equal the stitched contextual copy; `units[0]["unit_id"]` is that copy's id.
    - `test_synth_reports_phoneme_unit_ids`: word built from phonemes → `unit_id is None` and each phoneme span has the chosen phoneme copy's id.
    - `test_synth_skips_copy_whose_file_is_missing`: two copies, newest WAV deleted by hand → synth uses the older one; with both deleted → token in `skipped`.
- [ ] **Step 2: Run** `uv run pytest tests/test_resolver.py tests/test_synth.py -v` → FAIL.
- [ ] **Step 3: Implement.** Groups carry unit dicts; `audio.concatenate` receives `store.unit_path(u["id"])` lists. `_compute_units` reads durations from the files, as today.
- [ ] **Step 4: Run** `uv run pytest` → resolver/synth pass. `test_app.py` may fail until Task 4; note the count.
- [ ] **Step 5: Commit**: "Synth: choose word copies by neighbours and phrase position".

---

### Task 4: Routes on the store; remove the old file-layout API

**Files:**
- Modify: `app.py`, `audio.py` (add `encode_wav(samples: np.ndarray) -> bytes`), `inventory.py` (delete everything except `INVENTORY_DIR`, `normalise`, `is_valid_id`)
- Test: `tests/test_app.py` (seed via `store.add_unit`/`store.save_sentence`; assert via `store`), `tests/test_inventory.py` (keep only normalise/is_valid_id tests; variant tests are superseded by `test_store.py`)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: the HTTP API below. The frontend (Task 5) is its only client.

| Route | Request | Response |
|---|---|---|
| `GET /api/inventory` | — | `{"words": store.list_labels("word")}` |
| `POST /api/record` | form `word`, `audio` | always appends; `200 {"word", "unit_id"}`. The 409/`overwrite` path is removed |
| `POST /api/record/phoneme` | form `phoneme`, `audio` | `200 {"phoneme", "unit_id"}` |
| `GET /api/phoneme-inventory` | — | unchanged shape |
| `GET /api/units?kind=&label=` | kind `word`/`phoneme` | `{"kind", "label", "units": [unit dicts]}`; 400 bad kind / empty label / unknown phoneme |
| `GET /api/units/<unit_id>/audio` | — | `audio/wav` file; 404 if `get_unit` is None |
| `DELETE /api/units/<unit_id>` | — | 204, or 404 (incl. invalid id) |
| `POST /api/extract/propose` | `{"unit_id"}` or `{"word"}` (most recent copy) | as today, plus `"unit_id"` |
| `POST /api/extract/save` | `{"unit_id"` or `"word", "spans", "save"}` | phoneme units with `source="word"`, `source_unit_id`, `source_word=<label>`, in-word prev/next. Returns `{"saved": [{"symbol", "unit_id"}], "skipped"}` |
| `POST /api/extract/trim` | `{"unit_id"` or `"word", "start_ms", "end_ms"}` | `{"word", "unit_id", "duration_ms"}` via `replace_unit_audio` |
| `POST /api/phoneme/propose` / `trim` | `{"symbol", "unit_id"?}` | as today with `variant_id` renamed `unit_id`; a unit that isn't `kind=phoneme, label=symbol` → 404 |
| `/api/sentence/record`, `/api/sentences`, `/api/sentence/audio` | unchanged | backed by `store` |
| `/api/synthesize` | unchanged | units carry `unit_id` (Task 3) |
| `GET/DELETE /api/phoneme/variants` | **removed** | |

Shared helper in `app.py`: `_read_upload_wav(file) -> tuple[np.ndarray | None, tuple | None]`, which does the format validation used by the three upload routes. Replace the five copies of the inline `wave.open(out, "wb")` blocks with `audio.encode_wav`.

- [ ] **Step 1: Port `tests/test_app.py`.** Replace direct file writes into `tmp_path` (`hello.wav`, `phonemes/…`, `sentences/…`) with `store` seeding, and file-existence assertions with `store` queries. Delete the 409/overwrite tests and add:
  - `test_record_existing_word_appends_copy`: two posts → `len(store.list_units("word","hello")) == 2`.
  - `test_units_list_and_audio`: GET list returns both copies; GET audio returns `audio/wav` bytes equal to the file.
  - `test_unit_routes_reject_bad_ids`: `DELETE /api/units/..%2F..%2Fhello`, `/api/units/nope/audio` → 404; a seeded word WAV is untouched.
  - `test_phoneme_trim_rejects_word_unit_id`: phoneme trim with a word copy's id → 404.
  - `test_extract_save_records_source_unit`: saved phonemes have `source_unit_id == <word copy id>`, `source_word == "hello"`, first prev None / last next None.
  - `test_extract_targets_named_copy`: two `hello` copies; propose/trim with the older `unit_id` change only that one.
  - `test_delete_last_word_copy_keeps_cut_phonemes`: extract phonemes from the only `hello` copy, delete it → `/api/inventory` has no `hello`; phoneme copies still listed with `source_word == "hello"`.
  - `test_synthesize_units_carry_unit_id`.
- [ ] **Step 2: Run** `uv run pytest tests/test_app.py -v` → FAIL on the new/ported tests.
- [ ] **Step 3: Implement routes, `audio.encode_wav`, and strip `inventory.py`.** `grep -rn "inventory\.\(path\|has\|save\|list_words\|save_variant\|list_variants\|variant_path\|delete_variant\|best_phoneme_variant\|has_phoneme\|list_recorded_phonemes\|save_sentence\|find_sentence_by_text\|list_sentences\)" --include=*.py .` must return nothing.
- [ ] **Step 4: Run** `uv run pytest` → all pass.
- [ ] **Step 5: Commit**: "API: serve words and phonemes as unit copies from the store".

---

### Task 5: Frontend: copies panels, append-only recording, copy-targeted editor

**Files:**
- Modify: `static/app.js`, `static/style.css` (only if the panel styles are scoped to `.phoneme-chip`)

**Interfaces:**
- Consumes: the Task 4 API.

Changes:
- `togglePhonemeVariantsPanel(chip, symbol)` → `toggleUnitsPanel(chip, kind, label)`, used by phoneme chips and word chips (`refreshWordInventory`'s ✎ now opens the panel instead of the editor). Rows:
  - Meta text. Words: `isolated`, or `from sentence` with `prev=… next=…` and the position. Phonemes: `isolated`, or `from "<source_word>"` with `prev=… next=…`. Both end with `· N ms`.
  - ▶ plays `GET /api/units/<id>/audio`.
  - ✎ opens `openExtractEditor(label, id)` for words and `openPhonemeEditor(label, id)` for phonemes.
  - × confirms, `DELETE /api/units/<id>`, then refreshes inventory.
  - `playPhonemeVariant` is replaced by a generic `playUnit(id)`.
- `saveWord` / `savePhoneme`: remove the 409 branches. After a word save, `openExtractEditor(saved, unit_id)`.
- `openExtractEditor(word, unitId = null)` sends `unit_id` when present and keeps `editor.unitId` (from the response). Trim and save-phonemes send `unit_id`; after a trim, reopen with the same `unitId`. The phoneme editor renames `phonemeVariantId` → `unitId` and sends `unit_id`.
- `renderRenderedWords(units)`: one chip per unit with non-null `unit_id`, labelled `token`, ✎ → `activateTab('train-tab'); openExtractEditor(token, unit_id)`. `synthesise()` passes `units`; `compareSynth()` passes `normal.units`. Drop the `currentInventorySet`/`skipped` filtering it no longer needs.

- [ ] **Step 1: Implement the changes above.**
- [ ] **Step 2: Run a second dev server against a temp inventory and verify in the browser** (driving it with the browser tools). `INVENTORY_DIR` is relative to the working directory, so start it from an empty scratch dir: `uv run --project <repo> flask --app <repo>/app.py run --port 5001`. Seed audio by POSTing synthetic WAVs to `/api/record` and `/api/record/phoneme`.
  - Record `hello` twice → word chip ✎ shows 2 rows, ▶ plays each.
  - ✎ on the older row → editor trims only that copy (row duration changes, other row unchanged).
  - Save phonemes from it → `HH` panel shows a `from "hello"` row.
  - × on both `hello` rows → chip disappears; `HH` row remains.
  - Speak: synth `hello hello` with one copy → 2 chips; ✎ opens the editor on that copy.
  - No console errors.
- [ ] **Step 3: Run** `uv run pytest` → all pass.
- [ ] **Step 4: Commit**: "UI: copies panels for words and phonemes, edit the copy you heard".

---

### Task 6: Dump script, docs, and migrating the real inventory

**Files:**
- Create: `scripts/dump_db.py`
- Modify: `CLAUDE.md` (architecture: `store.py`, `migrate.py`, layout, resolver `Context`, test seeding via `store`), `docs/superpowers/specs/2026-11-07-sqlite-unit-store-design.md` (status → Implemented)

- [ ] **Step 1: `scripts/dump_db.py`**: prints sentences (`id  text`), then units grouped by kind and label (`id  source  prev/next  position  duration_ms  [missing]`). Usage: `uv run python scripts/dump_db.py`.
- [ ] **Step 2: Back up the real inventory.** Stop the dev server and copy `inventory/` to the session scratchpad. Report the backup path.
- [ ] **Step 3: Start the dev server, trigger migration, and verify.**
  - The log line reads `migrate: words=86 phonemes=39 sentences=2`.
  - `dump_db.py` lists 39 distinct phoneme IDs, none `00000000T000000000000`.
  - `inventory/phonemes/` and `inventory/sentences/` are gone.
  - Synthesising the Alice sentence returns audio, and its reference spectrogram renders.
- [ ] **Step 4: Update CLAUDE.md and spec status; run** `uv run pytest`.
- [ ] **Step 5: Commit and push**: "Docs + dump script for the SQLite store".
