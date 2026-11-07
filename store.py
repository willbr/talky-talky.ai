"""SQLite-backed store for recorded units (word and phoneme copies) and
reference sentences.

Metadata lives in `<INVENTORY_DIR>/talky.db`; audio stays as WAV files in
`<INVENTORY_DIR>/audio/<id>.wav`. IDs are `%Y%m%dT%H%M%S%f` UTC timestamps,
unique across units and sentences, so lexicographic order is chronological.

Every path is derived from `inventory.INVENTORY_DIR` at call time so tests
can redirect storage with monkeypatch.
"""
from __future__ import annotations

import io
import logging
import sqlite3
import wave
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import inventory

log = logging.getLogger("store")

DB_FILENAME = "talky.db"
AUDIO_SUBDIR = "audio"
KINDS = ("word", "phoneme")
SOURCES = ("isolated", "word", "sentence")
POSITIONS = ("mid", "comma", "final")
SCHEMA_VERSION = 1
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
"""

UNIT_FIELDS = (
    "id", "kind", "label", "source", "source_unit_id", "source_sentence_id",
    "source_word", "prev", "next", "position", "duration_ms",
)

_warned_missing: set[str] = set()


# --- paths & connection -----------------------------------------------------

def db_path() -> Path:
    return Path(inventory.INVENTORY_DIR) / DB_FILENAME


def audio_dir() -> Path:
    return Path(inventory.INVENTORY_DIR) / AUDIO_SUBDIR


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))


def connect() -> sqlite3.Connection:
    """Open the database, creating it (and its schema) if absent."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    create_schema(conn)
    conn.commit()
    return conn


def _rel(path: Path) -> str:
    return path.relative_to(Path(inventory.INVENTORY_DIR)).as_posix()


def _abs(wav_path: str) -> Path:
    return Path(inventory.INVENTORY_DIR) / wav_path


# --- ids ----------------------------------------------------------------------

def new_id(
    reserved: set[str] | None = None,
    when: datetime | None = None,
    check_files: bool = True,
) -> str:
    """Mint a timestamp ID, bumping by 1 µs until it is free.

    Free means: not in `reserved`, and (when `check_files`) no
    `audio/<id>.wav` already exists. Every unit and sentence owns exactly one
    file in `audio/`, so the file check covers both tables.
    """
    t = when or datetime.now(timezone.utc)
    reserved = reserved or set()
    adir = audio_dir()
    while True:
        candidate = t.strftime(ID_FORMAT)
        taken = candidate in reserved or (check_files and (adir / f"{candidate}.wav").exists())
        if not taken:
            return candidate
        t += timedelta(microseconds=1)


def _wav_duration_ms(wav_bytes: bytes) -> int:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            return int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, EOFError):
        return 0


# --- units --------------------------------------------------------------------

def canonical_label(kind: str, label: str) -> str:
    if kind == "word":
        return inventory.normalise(label)
    if kind == "phoneme":
        return label.strip().upper()
    raise ValueError(f"unknown kind: {kind!r}")


def unit_path(unit_id: str) -> Path:
    return audio_dir() / f"{unit_id}.wav"


def add_unit(
    kind: str,
    label: str,
    wav_bytes: bytes,
    *,
    source: str,
    prev: str | None = None,
    next: str | None = None,
    position: str | None = None,
    source_unit_id: str | None = None,
    source_sentence_id: str | None = None,
    source_word: str | None = None,
) -> str:
    label = canonical_label(kind, label)
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r}")
    if position is not None and position not in POSITIONS:
        raise ValueError(f"unknown position: {position!r}")
    adir = audio_dir()
    adir.mkdir(parents=True, exist_ok=True)
    uid = new_id()
    path = unit_path(uid)
    path.write_bytes(wav_bytes)
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO units (id, kind, label, wav_path, duration_ms, source,"
            " source_unit_id, source_sentence_id, source_word, prev, next, position)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, kind, label, _rel(path), _wav_duration_ms(wav_bytes), source,
             source_unit_id, source_sentence_id, source_word, prev, next, position),
        )
    return uid


def _present(row: sqlite3.Row) -> bool:
    if _abs(row["wav_path"]).exists():
        return True
    if row["id"] not in _warned_missing:
        _warned_missing.add(row["id"])
        log.warning("store: audio file missing for unit %s (%s)", row["id"], row["wav_path"])
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
            "SELECT * FROM units WHERE kind = ? AND label = ? ORDER BY id",
            (kind, label),
        ).fetchall()
    return [_unit_dict(r) for r in rows if _present(r)]


def list_labels(kind: str) -> list[str]:
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT id, label, wav_path FROM units WHERE kind = ? ORDER BY label, id",
            (kind,),
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
    unit = get_unit(unit_id)
    if unit is None:
        raise KeyError(unit_id)
    duration_ms = _wav_duration_ms(wav_bytes)
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
) -> dict | None:
    """Pick the copy whose recorded context best matches the target.

    +1 per matching neighbour (both-null counts as a match); words also score
    +1 for a matching position. Newest ID wins ties.
    """
    units = list_units(kind, label)
    if not units:
        return None

    def score(u: dict) -> tuple[int, str]:
        s = (u["prev"] == prev) + (u["next"] == next)
        if kind == "word":
            s += u["position"] == position
        return (s, u["id"])

    return max(units, key=score)


# --- sentences ---------------------------------------------------------------

def save_sentence(text: str, wav_bytes: bytes) -> str:
    """Persist a (text, audio) reference pair. Returns the id."""
    audio_dir().mkdir(parents=True, exist_ok=True)
    sid = new_id()
    path = audio_dir() / f"{sid}.wav"
    path.write_bytes(wav_bytes)
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO sentences (id, text, wav_path, created_at) VALUES (?, ?, ?, ?)",
            (sid, text, _rel(path), datetime.now(timezone.utc).isoformat()),
        )
    return sid


def find_sentence_by_text(text: str) -> Path | None:
    """Most recent reference WAV whose text matches after normalisation."""
    needle = inventory.normalise(text)
    if not needle:
        return None
    with closing(connect()) as conn:
        rows = conn.execute("SELECT id, text, wav_path FROM sentences ORDER BY id DESC").fetchall()
    for r in rows:
        if inventory.normalise(r["text"]) == needle and _present(r):
            return _abs(r["wav_path"])
    return None


def list_sentences() -> list[dict]:
    with closing(connect()) as conn:
        rows = conn.execute("SELECT id, text, wav_path FROM sentences ORDER BY id").fetchall()
    return [{"id": r["id"], "text": r["text"]} for r in rows if _present(r)]
