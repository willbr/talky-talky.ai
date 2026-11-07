import io
import json
import os
import sqlite3
import wave
from contextlib import closing

import numpy as np
import pytest

import audio
import inventory
import migrate
import store

LEGACY = "00000000T000000000000"


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _wav(duration_seconds=0.2):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes((np.ones(n, dtype=np.int16) * 10000).tobytes())
    return buf.getvalue()


def _write(path, data, mtime=1_700_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _phoneme_variant(root, symbol, sidecar=None, vid=LEGACY, mtime=1_700_000_000):
    d = root / "phonemes" / symbol.lower()
    _write(d / f"{vid}.wav", _wav(), mtime)
    meta = {"id": vid, "symbol": symbol, "source": "isolated", "prev": None,
            "next": None, "duration_ms": 200}
    meta.update(sidecar or {})
    _write(d / f"{vid}.json", json.dumps(meta), mtime)


def test_migrate_words(temp_inventory):
    _write(temp_inventory / "hello.wav", _wav())
    _write(temp_inventory / "apple.wav", _wav())

    assert store.list_labels("word") == ["apple", "hello"]
    assert not (temp_inventory / "hello.wav").exists()
    assert not (temp_inventory / "apple.wav").exists()
    assert len(list((temp_inventory / "audio").glob("*.wav"))) == 2
    unit = store.list_units("word", "hello")[0]
    assert unit["source"] == "isolated"
    assert unit["prev"] is None and unit["next"] is None and unit["position"] is None
    assert unit["duration_ms"] == 200


def test_migrate_legacy_phonemes_sharing_an_id_all_survive(temp_inventory):
    for sym in ("AA", "HH", "S"):
        _phoneme_variant(temp_inventory, sym)

    assert store.list_labels("phoneme") == ["AA", "HH", "S"]
    ids = [store.list_units("phoneme", s)[0]["id"] for s in ("AA", "HH", "S")]
    assert len(set(ids)) == 3
    assert LEGACY not in ids
    assert all(inventory.is_valid_id(i) for i in ids)
    assert not (temp_inventory / "phonemes").exists()


def test_migrate_phoneme_sidecar_context(temp_inventory):
    _phoneme_variant(temp_inventory, "AH",
                     {"source": "hello", "prev": "HH", "next": "L"},
                     vid="20260521T101010000000")
    unit = store.list_units("phoneme", "AH")[0]
    assert unit["source"] == "word"
    assert unit["source_word"] == "hello"
    assert unit["source_unit_id"] is None
    assert (unit["prev"], unit["next"]) == ("HH", "L")


def test_migrate_phoneme_without_sidecar_is_isolated(temp_inventory):
    _write(temp_inventory / "phonemes" / "aa" / "20260521T101010000000.wav", _wav())
    unit = store.list_units("phoneme", "AA")[0]
    assert unit["source"] == "isolated"


def test_migrate_flat_phoneme_file(temp_inventory):
    _write(temp_inventory / "phonemes" / "aa.wav", _wav())
    units = store.list_units("phoneme", "AA")
    assert len(units) == 1
    assert units[0]["source"] == "isolated"


def test_migrate_sentences_keep_ids(temp_inventory):
    sid = "20260521T194418044438"
    _write(temp_inventory / "sentences" / f"{sid}.wav", _wav())
    _write(temp_inventory / "sentences" / f"{sid}.txt", "Hello, world!")

    assert store.list_sentences() == [{"id": sid, "text": "Hello, world!"}]
    assert store.find_sentence_by_text("hello world") == temp_inventory / "audio" / f"{sid}.wav"
    assert not (temp_inventory / "sentences").exists()


def test_migrate_sentence_wav_without_text_is_left_in_place(temp_inventory):
    orphan = temp_inventory / "sentences" / "20260521T194418044438.wav"
    _write(orphan, _wav())
    assert store.list_sentences() == []
    assert orphan.exists()


def test_migrate_runs_once(temp_inventory):
    _write(temp_inventory / "hello.wav", _wav())
    assert store.list_labels("word") == ["hello"]

    _write(temp_inventory / "late.wav", _wav())
    assert store.list_labels("word") == ["hello"]
    assert (temp_inventory / "late.wav").exists()


def test_migrate_recovers_from_stale_tmp(temp_inventory):
    (temp_inventory / "talky.db.tmp").write_bytes(b"not a database")
    _write(temp_inventory / "hello.wav", _wav())
    assert store.list_labels("word") == ["hello"]
    assert not (temp_inventory / "talky.db.tmp").exists()


def test_migrate_is_deterministic(temp_inventory):
    _write(temp_inventory / "hello.wav", _wav(), mtime=1_700_000_000)
    _write(temp_inventory / "apple.wav", _wav(), mtime=1_700_000_000)
    _phoneme_variant(temp_inventory, "AA")
    _phoneme_variant(temp_inventory, "HH")

    def run_once():
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.row_factory = sqlite3.Row
            store.create_schema(conn)
            migrate.migrate(conn, temp_inventory)
            return [r[0] for r in conn.execute("SELECT id FROM units ORDER BY kind, label")]

    first = run_once()
    assert len(first) == 4
    assert run_once() == first


def test_migrate_returns_counts_and_originals(temp_inventory):
    _write(temp_inventory / "hello.wav", _wav())
    _phoneme_variant(temp_inventory, "AA")
    sid = "20260521T194418044438"
    _write(temp_inventory / "sentences" / f"{sid}.wav", _wav())
    _write(temp_inventory / "sentences" / f"{sid}.txt", "hi")

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.row_factory = sqlite3.Row
        store.create_schema(conn)
        counts, originals = migrate.migrate(conn, temp_inventory)

    assert counts == {"words": 1, "phonemes": 1, "sentences": 1}
    names = sorted(p.name for p in originals)
    assert names == sorted(["hello.wav", f"{LEGACY}.wav", f"{LEGACY}.json",
                            f"{sid}.wav", f"{sid}.txt"])
    # migrate itself never deletes anything
    assert all(p.exists() for p in originals)


def test_no_migration_needed_for_empty_inventory(temp_inventory):
    assert store.list_labels("word") == []
    assert (temp_inventory / "talky.db").exists()
