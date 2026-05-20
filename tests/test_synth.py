import io

import numpy as np
import pytest

import audio
import inventory
import synth


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _make_recording(path, n_samples=2000, amplitude=10000):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    audio.write_wav(path, samples)


def test_tokenise_splits_and_lowercases():
    assert synth.tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    assert synth.tokenise("Hello, world!") == ["hello", "world"]
    assert synth.tokenise("Don't stop.") == ["dont", "stop"]


def test_tokenise_collapses_whitespace():
    assert synth.tokenise("  hello   world  ") == ["hello", "world"]


def test_tokenise_empty_returns_empty_list():
    assert synth.tokenise("") == []
    assert synth.tokenise("   ") == []
    assert synth.tokenise("...!?") == []


def test_synthesise_all_words_present(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")
    _make_recording(temp_inventory / "world.wav")

    wav_bytes, skipped = synth.synthesise("hello world")

    assert wav_bytes is not None
    assert skipped == []
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) > 0


def test_synthesise_some_missing(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")

    wav_bytes, skipped = synth.synthesise("hello mars")

    assert wav_bytes is not None
    assert skipped == ["mars"]


def test_synthesise_all_missing(temp_inventory):
    wav_bytes, skipped = synth.synthesise("mars venus")

    assert wav_bytes is None
    assert skipped == ["mars", "venus"]


def test_synthesise_case_and_punctuation_insensitive(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")
    _make_recording(temp_inventory / "world.wav")

    wav_bytes, skipped = synth.synthesise("Hello, WORLD!")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_empty_sentence(temp_inventory):
    wav_bytes, skipped = synth.synthesise("")
    assert wav_bytes is None
    assert skipped == []
