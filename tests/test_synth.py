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

    wav_bytes, skipped, _ = synth.synthesise("hello world")

    assert wav_bytes is not None
    assert skipped == []
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) > 0


def test_synthesise_some_missing(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")

    wav_bytes, skipped, _ = synth.synthesise("hello mars")

    assert wav_bytes is not None
    assert skipped == ["mars"]


def test_synthesise_all_missing(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("mars venus")

    assert wav_bytes is None
    assert skipped == ["mars", "venus"]


def test_synthesise_case_and_punctuation_insensitive(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")
    _make_recording(temp_inventory / "world.wav")

    wav_bytes, skipped, _ = synth.synthesise("Hello, WORLD!")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_empty_sentence(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("")
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

    wav_bytes, skipped, _ = synth.synthesise("hello")

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

    wav_bytes, skipped, _ = synth.synthesise("hello")

    assert wav_bytes is None
    assert skipped == ["hello"]


def test_synthesise_phoneme_fallback_word_not_in_cmudict(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("xyzqqqzz")
    assert wav_bytes is None
    assert skipped == ["xyzqqqzz"]


def test_synthesise_word_resolver_takes_precedence(temp_inventory):
    """If a word is recorded AND all its phonemes are recorded, use the word file."""
    word_samples = np.ones(audio.SAMPLE_RATE * 2, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    wav_bytes, skipped, _ = synth.synthesise("hello")
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

    wav_bytes, skipped, _ = synth.synthesise("hello world")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_token_directly(temp_inventory):
    """If the user literally types 'AA' as a token and the AA phoneme is recorded,
    PhonemeResolver in the chain picks it up — no CMUdict lookup needed."""
    _make_phoneme_recording(temp_inventory, "AA")

    wav_bytes, skipped, _ = synth.synthesise("AA")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phonemes_only_bypasses_word_recording(temp_inventory):
    """With phonemes_only=True, a recorded word is ignored and the synth
    stitches the phonemes instead — verified by output length differing from
    the single-word concatenation."""
    word_samples = np.ones(audio.SAMPLE_RATE * 2, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    word_only, _, _ = synth.synthesise("hello")
    phonemes_only, skipped, _ = synth.synthesise("hello", phonemes_only=True)

    assert skipped == []
    assert phonemes_only is not None
    assert phonemes_only != word_only


def test_synthesise_units_track_token_spans(temp_inventory):
    """Each rendered token gets a [start_ms, end_ms] span; spans are
    monotonic and the last end_ms matches the total audio duration."""
    _make_recording(temp_inventory / "hello.wav", n_samples=audio.SAMPLE_RATE // 2)
    _make_recording(temp_inventory / "world.wav", n_samples=audio.SAMPLE_RATE)

    wav_bytes, skipped, units = synth.synthesise("hello world")
    assert skipped == []
    assert [u["token"] for u in units] == ["hello", "world"]
    assert units[0]["start_ms"] == 0
    assert units[0]["end_ms"] == units[1]["start_ms"]
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    total_ms = int(len(loaded) * 1000 / audio.SAMPLE_RATE)
    assert abs(units[1]["end_ms"] - total_ms) <= 2


def test_synthesise_units_skip_dropped_tokens(temp_inventory):
    """Skipped tokens do not appear in `units`."""
    _make_recording(temp_inventory / "hello.wav")
    wav_bytes, skipped, units = synth.synthesise("hello xyzqqqzz")
    assert skipped == ["xyzqqqzz"]
    assert [u["token"] for u in units] == ["hello"]


def test_synthesise_units_word_recording_has_no_phoneme_spans(temp_inventory):
    """A token rendered from a word recording exposes an empty phoneme list
    — we have no boundary information inside the word."""
    _make_recording(temp_inventory / "hello.wav")
    _, _, units = synth.synthesise("hello")
    assert units[0]["phonemes"] == []


def test_synthesise_units_phoneme_path_exposes_sub_spans(temp_inventory):
    """A token rendered via the phoneme path gets a sub-span per phoneme,
    contiguous and matching the parent unit's overall span."""
    _seed_all_phonemes_for_word(temp_inventory, "hello")
    _, _, units = synth.synthesise("hello")
    assert len(units) == 1
    word = units[0]
    phs = _phonemes.word_to_phonemes("hello")
    assert [p["symbol"] for p in word["phonemes"]] == phs
    assert word["phonemes"][0]["start_ms"] == word["start_ms"]
    assert word["phonemes"][-1]["end_ms"] == word["end_ms"]
    for prev, curr in zip(word["phonemes"], word["phonemes"][1:]):
        assert prev["end_ms"] == curr["start_ms"]


def test_synthesise_phonemes_only_skips_when_phoneme_missing(temp_inventory):
    """phonemes_only ignores the word file, so a missing phoneme means the
    word is skipped even if a word recording exists."""
    word_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    # Deliberately do NOT seed any phonemes.

    wav_bytes, skipped, _ = synth.synthesise("hello", phonemes_only=True)

    assert wav_bytes is None
    assert skipped == ["hello"]
