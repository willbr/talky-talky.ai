"""SQLite-backed store for recorded units (word and phoneme copies),
recordings (reference takes of any length) and their alignments.

Metadata lives in `<INVENTORY_DIR>/talky.db`; audio stays as WAV files in
`<INVENTORY_DIR>/audio/<id>.wav`. IDs are `%Y%m%dT%H%M%S%f` UTC timestamps,
unique across units and recordings, so lexicographic order is chronological.

Every path is derived from `inventory.INVENTORY_DIR` at call time.
"""
from __future__ import annotations

import io
import logging
import sqlite3
import wave
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import inventory

log = logging.getLogger("store")

DB_FILENAME = "talky.db"
AUDIO_SUBDIR = "audio"
KINDS = ("word", "phoneme")
SOURCES = ("isolated", "word", "sentence")
POSITIONS = ("mid", "comma", "final")
STATUSES = ("proposed", "saved", "excluded")
SCHEMA_VERSION = 2
ID_FORMAT = "%Y%m%dT%H%M%S%f"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS units (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('word','phoneme')),
  label TEXT NOT NULL,
  wav_path TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('isolated','word','sentence')),
  source_unit_id TEXT,
  source_recording_id TEXT,
  source_word TEXT,
  prev TEXT,
  next TEXT,
  position TEXT CHECK (position IN ('mid','comma','final')),
  source_sentence_idx INTEGER
);
CREATE INDEX IF NOT EXISTS units_kind_label ON units(kind, label);
CREATE INDEX IF NOT EXISTS units_source_recording ON units(source_recording_id, source_sentence_idx);
CREATE TABLE IF NOT EXISTS recordings (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, wav_path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alignment_segments (
  recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
  tier TEXT NOT NULL CHECK (tier IN ('sentence','word','phoneme')),
  sentence_idx INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  label TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  parent_idx INTEGER,
  edited INTEGER NOT NULL DEFAULT 0,
  status TEXT CHECK (status IN ('proposed','saved','excluded')),
  PRIMARY KEY (recording_id, tier, sentence_idx, idx)
);
"""

UNIT_FIELDS = (
    "id", "kind", "label", "source", "source_unit_id", "source_recording_id",
    "source_sentence_idx", "source_word", "prev", "next", "position", "duration_ms",
)

_warned_missing: set[str] = set()


# --- paths & connection -----------------------------------------------------

def db_path() -> Path:
    return Path(inventory.INVENTORY_DIR) / DB_FILENAME


def audio_dir() -> Path:
    return Path(inventory.INVENTORY_DIR) / AUDIO_SUBDIR


def _schema_version(conn: sqlite3.Connection) -> int | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if not exists:
        return None
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return row[0] if row else None


def _upgrade_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: `sentences` become `recordings` (takes of any length), units
    learn which sentence of a recording they came from, and alignment
    segments gain a sentence tier. v1's alignment table was always empty."""
    conn.executescript("""
        DROP TABLE IF EXISTS alignment_segments;
        ALTER TABLE sentences RENAME TO recordings;
        ALTER TABLE units RENAME COLUMN source_sentence_id TO source_recording_id;
        ALTER TABLE units ADD COLUMN source_sentence_idx INTEGER;
        UPDATE schema_version SET version = 2;
    """)
    log.info("store: upgraded database schema v1 -> v2")


def create_schema(conn: sqlite3.Connection) -> None:
    version = _schema_version(conn)
    if version == 1:
        _upgrade_v1_to_v2(conn)
    conn.executescript(SCHEMA)
    if version is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _create_with_migration(path: Path) -> None:
    """Build a new database at `path`, importing any legacy files.

    Built in `talky.db.tmp` and renamed into place only after the migration
    commits, so a crash leaves the legacy files intact and the next open
    starts over. Originals are deleted after the rename.
    """
    import migrate

    root = path.parent
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    with closing(_open(tmp)) as conn:
        create_schema(conn)
        counts, originals = migrate.migrate(conn, root)
        conn.commit()
    tmp.replace(path)
    migrate.cleanup(root, originals)
    if any(counts.values()):
        log.info("migrate: words=%d phonemes=%d recordings=%d",
                 counts["words"], counts["phonemes"], counts["sentences"])


_checked: set[Path] = set()


def connect() -> sqlite3.Connection:
    """Open the database, creating/migrating/upgrading it as needed."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _create_with_migration(path)
    conn = _open(path)
    conn.execute("PRAGMA foreign_keys = ON")
    if path not in _checked:
        _checked.add(path)
        create_schema(conn)
        conn.commit()
        import migrate
        if migrate.has_legacy_files(path.parent):
            log.warning("store: legacy inventory files remain beside %s; they are ignored", path)
    return conn


def _rel(path: Path) -> str:
    return path.relative_to(Path(inventory.INVENTORY_DIR)).as_posix()


def _abs(wav_path: str) -> Path:
    return Path(inventory.INVENTORY_DIR) / wav_path


# --- ids & audio files ------------------------------------------------------------

def new_id(
    reserved: set[str] | None = None,
    when: datetime | None = None,
    check_files: bool = True,
) -> str:
    """Mint a timestamp ID, bumping by 1 µs until it is free (not reserved and,
    when `check_files`, no `audio/<id>.wav` yet)."""
    t = when or datetime.now(timezone.utc)
    reserved = reserved or set()
    adir = audio_dir()
    while True:
        candidate = t.strftime(ID_FORMAT)
        taken = candidate in reserved or (check_files and (adir / f"{candidate}.wav").exists())
        if not taken:
            return candidate
        t += timedelta(microseconds=1)


def wav_duration_ms(wav_bytes: bytes) -> int:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            return int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, EOFError):
        return 0


def write_unit_wav(wav_bytes: bytes) -> str:
    """Mint an id and write `audio/<id>.wav`. The caller inserts the row."""
    audio_dir().mkdir(parents=True, exist_ok=True)
    uid = new_id()
    unit_path(uid).write_bytes(wav_bytes)
    return uid


# --- units ----------------------------------------------------------------------

def canonical_label(kind: str, label: str) -> str:
    if kind == "word":
        return inventory.normalise(label)
    if kind == "phoneme":
        return label.strip().upper()
    raise ValueError(f"unknown kind: {kind!r}")


def unit_path(unit_id: str) -> Path:
    return audio_dir() / f"{unit_id}.wav"


_INSERT_UNIT = (
    "INSERT INTO units (id, kind, label, wav_path, duration_ms, source, source_unit_id,"
    " source_recording_id, source_sentence_idx, source_word, prev, next, position)"
    " VALUES (:id, :kind, :label, :wav_path, :duration_ms, :source, :source_unit_id,"
    " :source_recording_id, :source_sentence_idx, :source_word, :prev, :next, :position)"
)


def _unit_row(uid: str, kind: str, label: str, duration_ms: int, **fields) -> dict:
    source = fields.get("source")
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r}")
    position = fields.get("position")
    if position is not None and position not in POSITIONS:
        raise ValueError(f"unknown position: {position!r}")
    row = {k: fields.get(k) for k in (
        "source", "source_unit_id", "source_recording_id", "source_sentence_idx",
        "source_word", "prev", "next", "position")}
    row.update(id=uid, kind=kind, label=canonical_label(kind, label),
               wav_path=_rel(unit_path(uid)), duration_ms=duration_ms)
    return row


def add_unit(kind: str, label: str, wav_bytes: bytes, *, source: str, **fields) -> str:
    """Write a copy's WAV and row. `fields`: prev, next, position, source_unit_id,
    source_recording_id, source_sentence_idx, source_word."""
    uid = write_unit_wav(wav_bytes)
    row = _unit_row(uid, kind, label, wav_duration_ms(wav_bytes), source=source, **fields)
    with closing(connect()) as conn, conn:
        conn.execute(_INSERT_UNIT, row)
    return uid


def _present(row: sqlite3.Row) -> bool:
    if _abs(row["wav_path"]).exists():
        return True
    if row["id"] not in _warned_missing:
        _warned_missing.add(row["id"])
        log.warning("store: audio file missing for %s (%s)", row["id"], row["wav_path"])
    return False


def _unit_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in UNIT_FIELDS}


def get_unit(unit_id: str) -> dict | None:
    if not inventory.is_valid_id(unit_id):
        return None
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()
    if row is None or not _present(row):
        return None
    return _unit_dict(row)


def list_units(kind: str, label: str) -> list[dict]:
    label = canonical_label(kind, label)
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM units WHERE kind = ? AND label = ? ORDER BY id", (kind, label),
        ).fetchall()
    return [_unit_dict(r) for r in rows if _present(r)]


def list_labels(kind: str) -> list[str]:
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT id, label, wav_path FROM units WHERE kind = ? ORDER BY label, id", (kind,),
        ).fetchall()
    labels: list[str] = []
    for r in rows:
        if (not labels or labels[-1] != r["label"]) and _present(r):
            labels.append(r["label"])
    return labels


def delete_unit(unit_id: str) -> bool:
    if not inventory.is_valid_id(unit_id):
        return False
    with closing(connect()) as conn, conn:
        row = conn.execute("SELECT wav_path FROM units WHERE id = ?", (unit_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM units WHERE id = ?", (unit_id,))
    _abs(row["wav_path"]).unlink(missing_ok=True)
    return True


def replace_unit_audio(unit_id: str, wav_bytes: bytes) -> int:
    if get_unit(unit_id) is None:
        raise KeyError(unit_id)
    duration_ms = wav_duration_ms(wav_bytes)
    unit_path(unit_id).write_bytes(wav_bytes)
    with closing(connect()) as conn, conn:
        conn.execute("UPDATE units SET duration_ms = ? WHERE id = ?", (duration_ms, unit_id))
    return duration_ms


def best_unit(
    kind: str,
    label: str,
    *,
    prev: str | None = None,
    next: str | None = None,
    position: str | None = None,
    exclude: Callable[[dict], bool] | None = None,
) -> dict | None:
    """Pick the copy whose recorded context best matches the target.

    +1 per matching neighbour (both-null counts as a match); words also score
    +1 for a matching position. Newest ID wins ties. `exclude` drops copies.
    """
    units = list_units(kind, label)
    if exclude is not None:
        units = [u for u in units if not exclude(u)]
    if not units:
        return None

    def score(u: dict) -> tuple[int, str]:
        s = (u["prev"] == prev) + (u["next"] == next)
        if kind == "word":
            s += u["position"] == position
        return (s, u["id"])

    return max(units, key=score)


# --- recordings ------------------------------------------------------------------

def recording_path(rec_id: str) -> Path:
    return audio_dir() / f"{rec_id}.wav"


def save_recording(text: str, wav_bytes: bytes) -> str:
    """Persist a (text, audio) take of any length. Returns the id."""
    audio_dir().mkdir(parents=True, exist_ok=True)
    rid = new_id()
    path = recording_path(rid)
    path.write_bytes(wav_bytes)
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO recordings (id, text, wav_path, created_at) VALUES (?, ?, ?, ?)",
            (rid, text, _rel(path), datetime.now(timezone.utc).isoformat()),
        )
    return rid


def get_recording(rec_id: str) -> dict | None:
    if not inventory.is_valid_id(rec_id):
        return None
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    if row is None or not _present(row):
        return None
    try:
        with wave.open(str(_abs(row["wav_path"])), "rb") as r:
            duration_ms = int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, OSError):
        duration_ms = 0
    return {"id": row["id"], "text": row["text"], "duration_ms": duration_ms}


def list_recordings() -> list[dict]:
    with closing(connect()) as conn:
        rows = conn.execute("SELECT id, text, wav_path FROM recordings ORDER BY id").fetchall()
    return [{"id": r["id"], "text": r["text"]} for r in rows if _present(r)]


def find_recordings(text: str) -> list[dict]:
    """Recordings whose whole text, or one of whose sentences, matches `text`
    after normalisation. Newest first; `sentence_idx` is None for a
    whole-text match."""
    needle = inventory.normalise(text)
    if not needle:
        return []
    with closing(connect()) as conn:
        recs = conn.execute("SELECT id, text, wav_path FROM recordings ORDER BY id DESC").fetchall()
        sents = conn.execute(
            "SELECT recording_id, idx, label FROM alignment_segments WHERE tier = 'sentence'"
            " ORDER BY recording_id DESC, idx"
        ).fetchall()
    present = {r["id"] for r in recs if _present(r)}
    matches: list[dict] = []
    for r in recs:
        if r["id"] in present and inventory.normalise(r["text"]) == needle:
            matches.append({"recording_id": r["id"], "sentence_idx": None})
    whole = {m["recording_id"] for m in matches}
    for s in sents:
        rid = s["recording_id"]
        if rid in present and rid not in whole and inventory.normalise(s["label"]) == needle:
            matches.append({"recording_id": rid, "sentence_idx": s["idx"]})
    matches.sort(key=lambda m: m["recording_id"], reverse=True)
    return matches


# --- alignment ----------------------------------------------------------------------

def set_sentences(rec_id: str, rows: list[dict]) -> None:
    """Insert sentence-tier rows `{idx, label, start_ms, end_ms}` as proposed."""
    with closing(connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO alignment_segments (recording_id, tier, sentence_idx, idx, label,"
            " start_ms, end_ms, status) VALUES (?, 'sentence', ?, ?, ?, ?, ?, 'proposed')",
            [(rec_id, r["idx"], r["idx"], r["label"], int(r["start_ms"]), int(r["end_ms"]))
             for r in rows],
        )


def get_sentences(rec_id: str) -> list[dict]:
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT idx, label, start_ms, end_ms, status FROM alignment_segments"
            " WHERE recording_id = ? AND tier = 'sentence' ORDER BY idx", (rec_id,),
        ).fetchall()
    return [{"idx": r["idx"], "text": r["label"], "start_ms": r["start_ms"],
             "end_ms": r["end_ms"], "status": r["status"]} for r in rows]


def update_sentence(rec_id: str, idx: int, *, text: str | None = None,
                    start_ms: int | None = None, end_ms: int | None = None,
                    status: str | None = None) -> None:
    sets, args = [], []
    for col, val in (("label", text), ("start_ms", start_ms), ("end_ms", end_ms), ("status", status)):
        if val is not None:
            sets.append(f"{col} = ?")
            args.append(val)
    if not sets:
        return
    with closing(connect()) as conn, conn:
        conn.execute(
            f"UPDATE alignment_segments SET {', '.join(sets)}"
            " WHERE recording_id = ? AND tier = 'sentence' AND idx = ?",
            (*args, rec_id, idx),
        )


def get_segments(rec_id: str, sentence_idx: int) -> dict:
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT tier, idx, label, start_ms, end_ms, parent_idx, edited FROM alignment_segments"
            " WHERE recording_id = ? AND sentence_idx = ? AND tier IN ('word','phoneme')"
            " ORDER BY tier, idx", (rec_id, sentence_idx),
        ).fetchall()
    out: dict[str, list[dict]] = {"words": [], "phonemes": []}
    for r in rows:
        out["words" if r["tier"] == "word" else "phonemes"].append({
            "idx": r["idx"], "label": r["label"], "start_ms": r["start_ms"],
            "end_ms": r["end_ms"], "parent_idx": r["parent_idx"], "edited": bool(r["edited"]),
        })
    return out


def _sentence_unit_rows(conn, rec_id: str, sentence_idx: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, wav_path FROM units WHERE source_recording_id = ? AND source_sentence_idx = ?",
        (rec_id, sentence_idx),
    ).fetchall()


def replace_sentence(rec_id: str, sentence_idx: int, *, words: list[dict],
                     phonemes: list[dict], units: list[dict]) -> list[str]:
    """Atomically replace one sentence's segments and copies, and mark it saved.

    `units` are rows for `_INSERT_UNIT` whose WAVs were already written via
    `write_unit_wav`. Old copies' WAVs are unlinked after commit.
    """
    with closing(connect()) as conn, conn:
        old = _sentence_unit_rows(conn, rec_id, sentence_idx)
        conn.execute(
            "DELETE FROM units WHERE source_recording_id = ? AND source_sentence_idx = ?",
            (rec_id, sentence_idx),
        )
        conn.execute(
            "DELETE FROM alignment_segments WHERE recording_id = ? AND sentence_idx = ?"
            " AND tier IN ('word','phoneme')", (rec_id, sentence_idx),
        )
        seg_rows = [
            (rec_id, tier, sentence_idx, s["idx"], s["label"], int(s["start_ms"]),
             int(s["end_ms"]), s.get("parent_idx"), int(bool(s.get("edited"))))
            for tier, segs in (("word", words), ("phoneme", phonemes)) for s in segs
        ]
        conn.executemany(
            "INSERT INTO alignment_segments (recording_id, tier, sentence_idx, idx, label,"
            " start_ms, end_ms, parent_idx, edited) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            seg_rows,
        )
        conn.executemany(_INSERT_UNIT, units)
        if words:
            conn.execute(
                "UPDATE alignment_segments SET status = 'saved', start_ms = ?, end_ms = ?"
                " WHERE recording_id = ? AND tier = 'sentence' AND idx = ?",
                (int(words[0]["start_ms"]), int(words[-1]["end_ms"]), rec_id, sentence_idx),
            )
    for r in old:
        _abs(r["wav_path"]).unlink(missing_ok=True)
    return [u["id"] for u in units]


def unit_row_for(uid: str, kind: str, label: str, duration_ms: int, **fields) -> dict:
    """Public builder for rows passed to `replace_sentence`."""
    return _unit_row(uid, kind, label, duration_ms, **fields)


def clear_sentence_copies(rec_id: str, sentence_idx: int) -> int:
    with closing(connect()) as conn, conn:
        old = _sentence_unit_rows(conn, rec_id, sentence_idx)
        conn.execute(
            "DELETE FROM units WHERE source_recording_id = ? AND source_sentence_idx = ?",
            (rec_id, sentence_idx),
        )
        conn.execute(
            "DELETE FROM alignment_segments WHERE recording_id = ? AND sentence_idx = ?"
            " AND tier IN ('word','phoneme')", (rec_id, sentence_idx),
        )
    for r in old:
        _abs(r["wav_path"]).unlink(missing_ok=True)
    return len(old)
