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


import phonemes as _phonemes  # alias to avoid colliding with any local var


def _make_phoneme_recording(temp_inventory, symbol, n_samples=2000, amplitude=10000):
    phonemes_dir = temp_inventory / "phonemes"
    phonemes_dir.mkdir(exist_ok=True)
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    audio.write_wav(phonemes_dir / f"{symbol.lower()}.wav", samples)


def _seed_all_phonemes_for_word(temp_inventory, word):
    """Create a phoneme recording for every phoneme the given word requires."""
    (temp_inventory / "phonemes").mkdir(exist_ok=True)
    phs = _phonemes.word_to_phonemes(word)
    assert phs is not None, f"{word!r} must be in CMUdict for this test"
    for ph in set(phs):
        _make_phoneme_recording(temp_inventory, ph)


def test_synthesise_phoneme_fallback_full_coverage(temp_inventory):
    """No word file for 'hello', but every phoneme recorded → synthesised, no skips."""
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    wav_bytes, skipped = synth.synthesise("hello")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_fallback_partial_missing(temp_inventory):
    """Word would be phonemise-able but one phoneme is missing → whole word skipped."""
    phs = _phonemes.word_to_phonemes("hello")
    assert phs is not None
    # Record every phoneme EXCEPT the first one.
    (temp_inventory / "phonemes").mkdir(exist_ok=True)
    for ph in set(phs[1:]):
        _make_phoneme_recording(temp_inventory, ph)

    wav_bytes, skipped = synth.synthesise("hello")

    assert wav_bytes is None
    assert skipped == ["hello"]


def test_synthesise_phoneme_fallback_word_not_in_cmudict(temp_inventory):
    wav_bytes, skipped = synth.synthesise("xyzqqqzz")
    assert wav_bytes is None
    assert skipped == ["xyzqqqzz"]


def test_synthesise_word_resolver_takes_precedence(temp_inventory):
    """If a word is recorded AND all its phonemes are recorded, use the word file."""
    word_samples = np.ones(audio.SAMPLE_RATE * 2, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    wav_bytes, skipped = synth.synthesise("hello")
    assert wav_bytes is not None
    assert skipped == []

    # The output should match a single-word concatenate, not a phoneme stitch.
    expected = audio.concatenate([[temp_inventory / "hello.wav"]])
    assert wav_bytes == expected


def test_synthesise_mixed_word_and_phoneme_words(temp_inventory):
    """Sentence with one recorded word + one phoneme-only word: both synthesised."""
    word_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "world")

    wav_bytes, skipped = synth.synthesise("hello world")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_token_directly(temp_inventory):
    """If the user literally types 'AA' as a token and the AA phoneme is recorded,
    PhonemeResolver in the chain picks it up — no CMUdict lookup needed."""
    _make_phoneme_recording(temp_inventory, "AA")

    wav_bytes, skipped = synth.synthesise("AA")

    assert wav_bytes is not None
    assert skipped == []
