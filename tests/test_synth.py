import io
import wave

import numpy as np
import pytest

import audio
import inventory
import phonemes as _phonemes
import store
import synth


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _wav_bytes(n_samples=2000, amplitude=10000):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _word(label, n_samples=2000, amplitude=10000, **kw):
    kw.setdefault("source", "isolated")
    return store.add_unit("word", label, _wav_bytes(n_samples, amplitude), **kw)


def _phoneme(symbol, n_samples=2000, amplitude=10000, **kw):
    kw.setdefault("source", "isolated")
    return store.add_unit("phoneme", symbol, _wav_bytes(n_samples, amplitude), **kw)


def _seed_all_phonemes_for_word(word, **kw):
    phs = _phonemes.word_to_phonemes(word)
    assert phs is not None, f"{word!r} must be in CMUdict for this test"
    return {ph: _phoneme(ph, **kw) for ph in set(phs)}


# --- tokenising -------------------------------------------------------------

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


def test_positions_basic():
    assert synth.tokenise_with_positions("Alice thought, she wait.") == [
        ("alice", "mid"), ("thought", "comma"), ("she", "mid"), ("wait", "final"),
    ]


def test_positions_handle_detached_and_quoted_punctuation():
    assert synth.tokenise_with_positions('"Hello," she said.') == [
        ("hello", "comma"), ("she", "mid"), ("said", "final"),
    ]
    assert synth.tokenise_with_positions("wait , as") == [
        ("wait", "comma"), ("as", "final"),
    ]


def test_positions_final_beats_comma_and_last_token_is_final():
    assert synth.tokenise_with_positions("stop,! go on; then") == [
        ("stop", "final"), ("go", "mid"), ("on", "comma"), ("then", "final"),
    ]


# --- basic synthesis ------------------------------------------------------------

def test_synthesise_all_words_present(temp_inventory):
    _word("hello")
    _word("world")
    wav_bytes, skipped, _ = synth.synthesise("hello world")
    assert wav_bytes is not None
    assert skipped == []
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) > 0


def test_synthesise_some_missing(temp_inventory):
    _word("hello")
    wav_bytes, skipped, _ = synth.synthesise("hello mars")
    assert wav_bytes is not None
    assert skipped == ["mars"]


def test_synthesise_all_missing(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("mars venus")
    assert wav_bytes is None
    assert skipped == ["mars", "venus"]


def test_synthesise_case_and_punctuation_insensitive(temp_inventory):
    _word("hello")
    _word("world")
    wav_bytes, skipped, _ = synth.synthesise("Hello, WORLD!")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_empty_sentence(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("")
    assert wav_bytes is None
    assert skipped == []


# --- phoneme fallback -----------------------------------------------------------

def test_synthesise_phoneme_fallback_full_coverage(temp_inventory):
    _seed_all_phonemes_for_word("hello")
    wav_bytes, skipped, _ = synth.synthesise("hello")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_fallback_partial_missing(temp_inventory):
    phs = _phonemes.word_to_phonemes("hello")
    for ph in set(phs[1:]) - {phs[0]}:
        _phoneme(ph)
    wav_bytes, skipped, _ = synth.synthesise("hello")
    assert wav_bytes is None
    assert skipped == ["hello"]


def test_synthesise_phoneme_fallback_word_not_in_cmudict(temp_inventory):
    wav_bytes, skipped, _ = synth.synthesise("xyzqqqzz")
    assert wav_bytes is None
    assert skipped == ["xyzqqqzz"]


def test_synthesise_word_resolver_takes_precedence(temp_inventory):
    uid = _word("hello", n_samples=audio.SAMPLE_RATE * 2, amplitude=5000)
    _seed_all_phonemes_for_word("hello")
    wav_bytes, skipped, _ = synth.synthesise("hello")
    assert skipped == []
    assert wav_bytes == audio.concatenate([[store.unit_path(uid)]])


def test_synthesise_mixed_word_and_phoneme_words(temp_inventory):
    _word("hello", n_samples=audio.SAMPLE_RATE, amplitude=5000)
    _seed_all_phonemes_for_word("world")
    wav_bytes, skipped, _ = synth.synthesise("hello world")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_token_directly(temp_inventory):
    _phoneme("AA")
    wav_bytes, skipped, _ = synth.synthesise("AA")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phonemes_only_bypasses_word_recording(temp_inventory):
    _word("hello", n_samples=audio.SAMPLE_RATE * 2, amplitude=5000)
    _seed_all_phonemes_for_word("hello")
    word_only, _, _ = synth.synthesise("hello")
    phonemes_only, skipped, _ = synth.synthesise("hello", phonemes_only=True)
    assert skipped == []
    assert phonemes_only is not None
    assert phonemes_only != word_only


def test_synthesise_phonemes_only_skips_when_phoneme_missing(temp_inventory):
    _word("hello", n_samples=audio.SAMPLE_RATE, amplitude=5000)
    wav_bytes, skipped, _ = synth.synthesise("hello", phonemes_only=True)
    assert wav_bytes is None
    assert skipped == ["hello"]


# --- unit spans -------------------------------------------------------------

def test_synthesise_units_track_token_spans(temp_inventory):
    _word("hello", n_samples=audio.SAMPLE_RATE // 2)
    _word("world", n_samples=audio.SAMPLE_RATE)
    wav_bytes, skipped, units = synth.synthesise("hello world")
    assert skipped == []
    assert [u["token"] for u in units] == ["hello", "world"]
    assert units[0]["start_ms"] == 0
    assert units[0]["end_ms"] == units[1]["start_ms"]
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    total_ms = int(len(loaded) * 1000 / audio.SAMPLE_RATE)
    assert abs(units[1]["end_ms"] - total_ms) <= 2


def test_synthesise_units_skip_dropped_tokens(temp_inventory):
    _word("hello")
    _, skipped, units = synth.synthesise("hello xyzqqqzz")
    assert skipped == ["xyzqqqzz"]
    assert [u["token"] for u in units] == ["hello"]


def test_synthesise_units_word_recording_has_no_phoneme_spans(temp_inventory):
    _word("hello")
    _, _, units = synth.synthesise("hello")
    assert units[0]["phonemes"] == []


def test_synthesise_units_phoneme_path_exposes_sub_spans(temp_inventory):
    _seed_all_phonemes_for_word("hello")
    _, _, units = synth.synthesise("hello")
    assert len(units) == 1
    word = units[0]
    assert [p["symbol"] for p in word["phonemes"]] == _phonemes.word_to_phonemes("hello")
    assert word["phonemes"][0]["start_ms"] == word["start_ms"]
    assert word["phonemes"][-1]["end_ms"] == word["end_ms"]
    for prev, curr in zip(word["phonemes"], word["phonemes"][1:]):
        assert prev["end_ms"] == curr["start_ms"]


# --- copy selection -----------------------------------------------------------

def test_phoneme_selection_uses_context_match(temp_inventory):
    phs = _phonemes.word_to_phonemes("hello")
    _seed_all_phonemes_for_word("hello", amplitude=1000)
    i = phs.index("HH")
    contextual = _phoneme(
        "HH", source="word", source_word="hello",
        prev=phs[i - 1] if i > 0 else None,
        next=phs[i + 1] if i < len(phs) - 1 else None,
        amplitude=15000,
    )
    wav_bytes, skipped, units = synth.synthesise("hello")
    assert skipped == []
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert int(np.max(np.abs(loaded))) >= 14000
    assert units[0]["phonemes"][i]["unit_id"] == contextual


def test_synth_picks_word_copy_by_context(temp_inventory):
    _word("hello", amplitude=1000)                     # isolated, older
    contextual = _word("hello", amplitude=15000, source="sentence",
                       prev=None, next="world", position="mid")
    _word("hello", amplitude=2000)                     # isolated, newest
    world = _word("world")
    wav_bytes, skipped, units = synth.synthesise("hello world")
    assert skipped == []
    assert units[0]["unit_id"] == contextual
    assert units[1]["unit_id"] == world
    expected = audio.concatenate([[store.unit_path(contextual)], [store.unit_path(world)]])
    assert wav_bytes == expected


def test_synth_reports_phoneme_unit_ids(temp_inventory):
    ids = _seed_all_phonemes_for_word("hello")
    _, _, units = synth.synthesise("hello")
    assert units[0]["unit_id"] is None
    for span in units[0]["phonemes"]:
        assert span["unit_id"] == ids[span["symbol"]]


def test_synth_direct_phoneme_token_reports_unit_id(temp_inventory):
    uid = _phoneme("AA")
    _, _, units = synth.synthesise("AA")
    assert units[0]["unit_id"] is None
    assert units[0]["phonemes"][0]["unit_id"] == uid


def test_synth_skips_copy_whose_file_is_missing(temp_inventory):
    older = _word("hello")
    newer = _word("hello")
    store.unit_path(newer).unlink()
    _, skipped, units = synth.synthesise("hello")
    assert skipped == []
    assert units[0]["unit_id"] == older

    store.unit_path(older).unlink()
    wav_bytes, skipped, _ = synth.synthesise("hello")
    assert wav_bytes is None
    assert skipped == ["hello"]
