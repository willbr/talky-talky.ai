import logging

import numpy as np

import audio
import extraction
import phonemes


def _tone(duration_ms: int, amplitude: int = 10000) -> np.ndarray:
    n = int(audio.SAMPLE_RATE * duration_ms / 1000)
    return np.ones(n, dtype=np.int16) * amplitude


def _silence(duration_ms: int) -> np.ndarray:
    n = int(audio.SAMPLE_RATE * duration_ms / 1000)
    return np.zeros(n, dtype=np.int16)


def test_duration_table_covers_every_phoneme():
    assert set(extraction.PHONEME_DURATION_MS.keys()) == phonemes.PHONEMES_SET
    assert all(isinstance(v, int) and v > 0 for v in extraction.PHONEME_DURATION_MS.values())


def test_propose_boundaries_basic_returns_n_plus_one_monotonic():
    samples = _tone(800)
    bounds = extraction.propose_boundaries(samples, ["HH", "AH", "L", "OW"])
    assert len(bounds) == 5
    for i in range(len(bounds) - 1):
        assert bounds[i] <= bounds[i + 1]


def test_propose_boundaries_proportional_to_priors():
    # 460 ms of loud audio, AA (150) + K (80) -> total prior 230, span 460 -> first split at 300 ms.
    samples = _tone(460)
    bounds = extraction.propose_boundaries(samples, ["AA", "K"])
    assert bounds[0] == 0
    assert bounds[2] == 460
    vowel_width = bounds[1] - bounds[0]
    stop_width = bounds[2] - bounds[1]
    assert abs(vowel_width - 300) <= 5
    assert abs(stop_width - 160) <= 5


def test_propose_boundaries_skips_leading_silence():
    samples = np.concatenate([_silence(200), _tone(600)])
    bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert 190 <= bounds[0] <= 210
    assert bounds[-1] == 800


def test_propose_boundaries_skips_trailing_silence():
    samples = np.concatenate([_tone(600), _silence(300)])
    bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert bounds[0] == 0
    assert 590 <= bounds[-1] <= 610


def test_propose_boundaries_all_silent_fallback(caplog):
    samples = _silence(500)
    with caplog.at_level(logging.WARNING, logger="extraction"):
        bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert bounds[0] == 0
    assert bounds[-1] == 500
    assert any("silent" in r.message.lower() for r in caplog.records)


def test_propose_boundaries_unknown_phoneme_uses_default(caplog):
    samples = _tone(400)
    with caplog.at_level(logging.WARNING, logger="extraction"):
        bounds = extraction.propose_boundaries(samples, ["FAKE", "AH"])
    assert len(bounds) == 3
    assert any("unknown phoneme" in r.message.lower() for r in caplog.records)


def test_propose_boundaries_empty_sequence_returns_single_zero():
    samples = _tone(400)
    assert extraction.propose_boundaries(samples, []) == [0]
