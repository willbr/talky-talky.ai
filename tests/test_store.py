import io
import wave
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import audio
import inventory
import store


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _wav(duration_seconds=0.2, amplitude=10000):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    samples = np.ones(n, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def test_add_unit_writes_wav_and_row(temp_inventory):
    uid = store.add_unit("word", "Hello!", _wav(0.2), source="isolated")
    assert inventory.is_valid_id(uid)
    assert store.unit_path(uid) == temp_inventory / "audio" / f"{uid}.wav"
    assert store.unit_path(uid).exists()
    unit = store.get_unit(uid)
    assert unit["label"] == "hello"
    assert unit["kind"] == "word"
    assert unit["source"] == "isolated"
    assert unit["duration_ms"] == 200
    assert (temp_inventory / "talky.db").exists()


def test_phoneme_label_is_uppercased(temp_inventory):
    uid = store.add_unit("phoneme", "aa", _wav(), source="isolated")
    assert store.get_unit(uid)["label"] == "AA"
    assert [u["id"] for u in store.list_units("phoneme", "Aa")] == [uid]


def test_list_units_oldest_first(temp_inventory):
    a = store.add_unit("word", "hello", _wav(), source="isolated")
    b = store.add_unit("word", "hello", _wav(), source="isolated")
    assert [u["id"] for u in store.list_units("word", "hello")] == [a, b]


def test_list_labels_sorted_distinct(temp_inventory):
    store.add_unit("word", "hello", _wav(), source="isolated")
    store.add_unit("word", "hello", _wav(), source="isolated")
    store.add_unit("word", "apple", _wav(), source="isolated")
    store.add_unit("phoneme", "AA", _wav(), source="isolated")
    assert store.list_labels("word") == ["apple", "hello"]
    assert store.list_labels("phoneme") == ["AA"]


def test_new_id_bumps_on_collision(temp_inventory):
    when = datetime(2026, 11, 7, 12, 0, 0, 0, tzinfo=timezone.utc)
    id0 = when.strftime("%Y%m%dT%H%M%S%f")
    assert store.new_id(when=when) == id0
    bumped = (when + timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S%f")
    assert store.new_id(reserved={id0}, when=when) == bumped


def test_new_id_avoids_existing_audio_files(temp_inventory):
    when = datetime(2026, 11, 7, 12, 0, 0, 0, tzinfo=timezone.utc)
    id0 = when.strftime("%Y%m%dT%H%M%S%f")
    (temp_inventory / "audio").mkdir()
    (temp_inventory / "audio" / f"{id0}.wav").write_bytes(b"x")
    assert store.new_id(when=when) != id0
    assert store.new_id(when=when, check_files=False) == id0


def test_delete_unit(temp_inventory):
    uid = store.add_unit("word", "hello", _wav(), source="isolated")
    assert store.delete_unit(uid) is True
    assert not store.unit_path(uid).exists()
    assert store.get_unit(uid) is None
    assert store.delete_unit(uid) is False


def test_delete_unit_rejects_malformed_id(temp_inventory):
    (temp_inventory / "hello.wav").write_bytes(b"keep")
    for bad in ["../hello", "..\\hello", "", "nope"]:
        assert store.delete_unit(bad) is False
    assert (temp_inventory / "hello.wav").exists()


def test_replace_unit_audio_refreshes_duration(temp_inventory):
    uid = store.add_unit("word", "hello", _wav(0.5), source="isolated")
    assert store.replace_unit_audio(uid, _wav(0.1)) == 100
    assert store.get_unit(uid)["duration_ms"] == 100
    samples, _ = audio.read_wav(store.unit_path(uid))
    assert len(samples) == int(audio.SAMPLE_RATE * 0.1)


def test_replace_unit_audio_unknown_raises(temp_inventory):
    with pytest.raises(KeyError):
        store.replace_unit_audio("20260101T000000000000", _wav())


def test_best_word_unit_scoring(temp_inventory):
    plain = store.add_unit("word", "wait", _wav(), source="isolated")
    full = store.add_unit("word", "wait", _wav(), source="sentence",
                          prev="she", next="as", position="comma")
    store.add_unit("word", "wait", _wav(), source="sentence",
                   prev="she", next=None, position="mid")
    best = store.best_unit("word", "wait", prev="she", next="as", position="comma")
    assert best["id"] == full
    assert store.best_unit("word", "wait")["id"] == plain


def test_best_unit_tiebreaks_to_newest(temp_inventory):
    store.add_unit("word", "wait", _wav(), source="isolated")
    newer = store.add_unit("word", "wait", _wav(), source="isolated")
    assert store.best_unit("word", "wait", prev="x", next="y", position="mid")["id"] == newer


def test_best_phoneme_unit_ignores_position(temp_inventory):
    store.add_unit("phoneme", "AA", _wav(), source="word", prev="HH", next="L",
                   position="final")
    newer = store.add_unit("phoneme", "AA", _wav(), source="word", prev="HH", next="L")
    assert store.best_unit("phoneme", "AA", prev="HH", next="L", position="final")["id"] == newer


def test_best_unit_none_when_empty(temp_inventory):
    assert store.best_unit("word", "nothing") is None


def test_missing_file_rows_are_skipped(temp_inventory):
    keep = store.add_unit("word", "hello", _wav(), source="isolated")
    gone = store.add_unit("word", "hello", _wav(), source="isolated")
    lonely = store.add_unit("word", "lonely", _wav(), source="isolated")
    store.unit_path(gone).unlink()
    store.unit_path(lonely).unlink()

    assert [u["id"] for u in store.list_units("word", "hello")] == [keep]
    assert store.list_labels("word") == ["hello"]
    assert store.best_unit("word", "hello")["id"] == keep
    assert store.get_unit(gone) is None
    assert store.best_unit("word", "lonely") is None

    with store.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    assert n == 3   # rows are kept, not auto-deleted


def test_sentence_roundtrip(temp_inventory):
    sid = store.save_sentence("Hello world", _wav())
    assert (temp_inventory / "audio" / f"{sid}.wav").exists()
    assert store.find_sentence_by_text("hello,  WORLD") == temp_inventory / "audio" / f"{sid}.wav"
    assert store.find_sentence_by_text("something else") is None
    assert store.list_sentences() == [{"id": sid, "text": "Hello world"}]


def test_find_sentence_prefers_latest(temp_inventory):
    store.save_sentence("hello", _wav())
    newer = store.save_sentence("Hello!", _wav())
    assert store.find_sentence_by_text("hello").stem == newer


def test_sentence_and_unit_ids_never_collide(temp_inventory):
    ids = {store.save_sentence("a", _wav()) for _ in range(5)}
    ids |= {store.add_unit("word", "a", _wav(), source="isolated") for _ in range(5)}
    assert len(ids) == 10
