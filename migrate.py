"""One-time migration from the legacy file layout into the SQLite store.

Legacy layout (pre-2026-11):
  <dir>/<word>.wav                          word recordings, one per word
  <dir>/phonemes/<sym>/<id>.{wav,json}      phoneme variants + sidecar
  <dir>/phonemes/<sym>.wav                  older flat phoneme files
  <dir>/sentences/<id>.{wav,txt}            reference recordings

`migrate()` copies audio into `<dir>/audio/<id>.wav` and inserts rows. It
never deletes anything; it returns the originals so the caller can remove
them after the database has been committed and moved into place.

Unit IDs are minted from each file's modification time, so re-running after
a crash produces the same IDs (and overwrites the same copies).
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import sqlite3
import wave
from datetime import datetime, timezone
from pathlib import Path

import store

log = logging.getLogger("migrate")


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def _duration_ms(path: Path) -> int:
    try:
        with wave.open(io.BytesIO(path.read_bytes()), "rb") as r:
            return int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, EOFError, OSError):
        return 0


def _read_sidecar(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def has_legacy_files(inventory_dir: Path) -> bool:
    return (
        any(inventory_dir.glob("*.wav"))
        or (inventory_dir / "phonemes").is_dir()
        or (inventory_dir / "sentences").is_dir()
    )


def migrate(conn: sqlite3.Connection, inventory_dir: Path) -> tuple[dict[str, int], list[Path]]:
    inventory_dir = Path(inventory_dir)
    audio = inventory_dir / store.AUDIO_SUBDIR
    reserved: set[str] = set()
    originals: list[Path] = []
    counts = {"words": 0, "phonemes": 0, "sentences": 0}

    def copy_in(src: Path, new: str) -> str:
        audio.mkdir(parents=True, exist_ok=True)
        dst = audio / f"{new}.wav"
        shutil.copy2(src, dst)
        return dst.relative_to(inventory_dir).as_posix()

    def insert_unit(src: Path, kind: str, label: str, **fields) -> None:
        uid = store.new_id(reserved, when=_mtime(src), check_files=False)
        reserved.add(uid)
        rel = copy_in(src, uid)
        conn.execute(
            "INSERT INTO units (id, kind, label, wav_path, duration_ms, source,"
            " source_unit_id, source_sentence_id, source_word, prev, next, position)"
            " VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, NULL)",
            (uid, kind, label, rel, _duration_ms(src), fields.get("source", "isolated"),
             fields.get("source_word"), fields.get("prev"), fields.get("next")),
        )

    # 1. Sentences keep their IDs; reserve them before minting unit IDs.
    sent_dir = inventory_dir / "sentences"
    if sent_dir.is_dir():
        for wav in sorted(sent_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                log.warning("migrate: %s has no .txt; leaving it in place", wav)
                continue
            sid = wav.stem
            try:
                created = datetime.strptime(sid, store.ID_FORMAT).replace(tzinfo=timezone.utc)
            except ValueError:
                sid = store.new_id(reserved, when=_mtime(wav), check_files=False)
                created = _mtime(wav)
            reserved.add(sid)
            rel = copy_in(wav, sid)
            conn.execute(
                "INSERT INTO sentences (id, text, wav_path, created_at) VALUES (?, ?, ?, ?)",
                (sid, txt.read_text(encoding="utf-8"), rel, created.isoformat()),
            )
            originals += [wav, txt]
            counts["sentences"] += 1

    # 2. Words, by label.
    for wav in sorted(inventory_dir.glob("*.wav")):
        label = store.canonical_label("word", wav.stem)
        if not label:
            log.warning("migrate: skipping word file with empty label: %s", wav)
            continue
        insert_unit(wav, "word", label)
        originals.append(wav)
        counts["words"] += 1

    # 3. Phonemes, by (symbol, old id). Flat files first within a symbol.
    ph_dir = inventory_dir / "phonemes"
    if ph_dir.is_dir():
        items: list[tuple[str, str, Path, Path | None]] = []
        for flat in ph_dir.glob("*.wav"):
            items.append((flat.stem.upper(), "", flat, None))
        for sym_dir in ph_dir.iterdir():
            if not sym_dir.is_dir():
                continue
            for wav in sym_dir.glob("*.wav"):
                items.append((sym_dir.name.upper(), wav.stem, wav, wav.with_suffix(".json")))
        for symbol, _old, wav, sidecar in sorted(items, key=lambda t: (t[0], t[1])):
            meta = _read_sidecar(sidecar) if sidecar and sidecar.exists() else {}
            src = meta.get("source")
            if isinstance(src, str) and src and src != "isolated":
                fields: dict[str, str | None] = {"source": "word", "source_word": src}
            else:
                fields = {"source": "isolated"}
            fields["prev"] = meta.get("prev")
            fields["next"] = meta.get("next")
            insert_unit(wav, "phoneme", symbol, **fields)
            originals.append(wav)
            if sidecar and sidecar.exists():
                originals.append(sidecar)
            counts["phonemes"] += 1

    return counts, originals


def cleanup(inventory_dir: Path, originals: list[Path]) -> None:
    """Delete migrated originals and the now-empty legacy folders."""
    for p in originals:
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("migrate: could not delete %s: %s", p, e)
    ph_dir = inventory_dir / "phonemes"
    if ph_dir.is_dir():
        for sub in ph_dir.iterdir():
            if sub.is_dir():
                try:
                    sub.rmdir()
                except OSError:
                    pass
    for d in (ph_dir, inventory_dir / "sentences"):
        try:
            d.rmdir()
        except OSError:
            pass
