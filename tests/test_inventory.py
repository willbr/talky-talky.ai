import io
import wave

import numpy as np
import pytest

import audio
import inventory


def _wav_bytes(duration_seconds=0.2, amplitude=10000):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    samples = np.ones(n, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def test_normalise_lowercases():
    assert inventory.normalise("Hello") == "hello"


def test_normalise_strips_punctuation():
    assert inventory.normalise("Hello!") == "hello"
    assert inventory.normalise("Don't") == "dont"
    assert inventory.normalise("(world)") == "world"


def test_normalise_strips_surrounding_whitespace():
    assert inventory.normalise("  hello  ") == "hello"


def test_normalise_empty_input_returns_empty():
    assert inventory.normalise("") == ""
    assert inventory.normalise("...") == ""
    assert inventory.normalise("!?") == ""


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def test_path_uses_normalised_filename(temp_inventory):
    assert inventory.path("Hello!") == temp_inventory / "hello.wav"


def test_save_writes_bytes_to_disk(temp_inventory):
    wav_bytes = b"RIFF....fakewav....data...."
    inventory.save("hello", wav_bytes)
    assert (temp_inventory / "hello.wav").read_bytes() == wav_bytes


def test_has_reflects_existence(temp_inventory):
    assert not inventory.has("hello")
    inventory.save("hello", b"x")
    assert inventory.has("hello")


def test_list_words_returns_sorted_unique(temp_inventory):
    inventory.save("world", b"x")
    inventory.save("hello", b"x")
    inventory.save("apple", b"x")
    assert inventory.list_words() == ["apple", "hello", "world"]


def test_save_creates_inventory_dir_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "newdir"
    monkeypatch.setattr(inventory, "INVENTORY_DIR", target)
    inventory.save("hello", b"x")
    assert target.is_dir()
    assert (target / "hello.wav").exists()


def test_save_variant_writes_wav_and_metadata(temp_inventory):
    wav = _wav_bytes()
    vid = inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)
    sym_dir = temp_inventory / "phonemes" / "aa"
    assert (sym_dir / f"{vid}.wav").read_bytes() == wav
    import json
    meta = json.loads((sym_dir / f"{vid}.json").read_text(encoding="utf-8"))
    assert meta["symbol"] == "AA"
    assert meta["source"] == "isolated"
    assert meta["prev"] is None
    assert meta["next"] is None
    assert meta["duration_ms"] > 0


def test_list_variants_chronological(temp_inventory):
    import time
    vid1 = inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    time.sleep(0.001)  # nudge timestamp granularity
    vid2 = inventory.save_variant("AA", _wav_bytes(), source="hello", prev="HH", next="L")
    listed = inventory.list_variants("AA")
    assert [v["id"] for v in listed] == sorted([vid1, vid2])


def test_list_variants_empty(temp_inventory):
    assert inventory.list_variants("AA") == []


def test_has_phoneme_reflects_variants(temp_inventory):
    assert not inventory.has_phoneme("AA")
    inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    assert inventory.has_phoneme("AA")


def test_variant_path_round_trip(temp_inventory):
    vid = inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    p = inventory.variant_path("AA", vid)
    assert p.exists() and p.suffix == ".wav"


def test_delete_variant_removes_pair(temp_inventory):
    vid = inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    assert inventory.delete_variant("AA", vid) is True
    assert not inventory.variant_path("AA", vid).exists()
    assert not (temp_inventory / "phonemes" / "aa" / f"{vid}.json").exists()
    assert inventory.delete_variant("AA", vid) is False


def test_delete_variant_rejects_malformed_id(temp_inventory):
    (temp_inventory / "hello.wav").write_bytes(_wav_bytes())
    for bad in ["../../hello", "..\\..\\hello", "", "nope", "20260521T211755123456/../x"]:
        assert inventory.delete_variant("AA", bad) is False
    assert (temp_inventory / "hello.wav").exists()


def test_best_phoneme_variant_prefers_full_context_match(temp_inventory):
    none_id = inventory.save_variant("HH", _wav_bytes(), source="isolated", prev=None, next=None)
    half_id = inventory.save_variant("HH", _wav_bytes(), source="hat", prev=None, next="AE")
    full_id = inventory.save_variant("HH", _wav_bytes(), source="hello", prev=None, next="AH")
    # Target context is prev=None, next=AH — matches full_id on both sides,
    # half_id on one side, none_id on prev only.
    assert inventory.best_phoneme_variant("HH", None, "AH") == inventory.variant_path("HH", full_id)


def test_best_phoneme_variant_tiebreaks_to_most_recent(temp_inventory):
    import time
    older = inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    time.sleep(0.001)
    newer = inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    # Both score equally for any target — newer wins.
    assert inventory.best_phoneme_variant("AA") == inventory.variant_path("AA", newer)
    assert older != newer  # sanity


def test_best_phoneme_variant_returns_none_when_empty(temp_inventory):
    assert inventory.best_phoneme_variant("AA") is None


def test_legacy_flat_file_is_migrated_once(temp_inventory):
    flat = temp_inventory / "phonemes" / "hh.wav"
    flat.parent.mkdir(parents=True, exist_ok=True)
    flat.write_bytes(_wav_bytes())

    listed = inventory.list_variants("HH")
    assert len(listed) == 1
    legacy = listed[0]
    assert legacy["id"] == inventory.LEGACY_VARIANT_ID
    assert legacy["source"] == "isolated"
    assert not flat.exists()  # flat file gone after migration

    # Second call must not re-migrate (it would crash trying to move a
    # non-existent flat file otherwise, but check the listing stays put).
    again = inventory.list_variants("HH")
    assert len(again) == 1
    assert again[0]["id"] == inventory.LEGACY_VARIANT_ID


def test_list_recorded_phonemes_returns_uppercase_sorted(temp_inventory):
    inventory.save_variant("OW", _wav_bytes(), source="isolated", prev=None, next=None)
    inventory.save_variant("AA", _wav_bytes(), source="isolated", prev=None, next=None)
    inventory.save_variant("K", _wav_bytes(), source="isolated", prev=None, next=None)
    assert inventory.list_recorded_phonemes() == ["AA", "K", "OW"]


def test_list_recorded_phonemes_empty_when_no_dir(temp_inventory):
    assert inventory.list_recorded_phonemes() == []
