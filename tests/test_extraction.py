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


def test_propose_boundaries_ignores_single_window_click():
    """A single very-brief loud spike (mic tap, mouse click) at the start or
    end of the recording must not be counted as the loud region's onset/offset."""
    # 50 ms quiet, 10 ms big spike, 150 ms quiet, 600 ms sustained tone, 200 ms quiet.
    quiet_lead = np.zeros(int(audio.SAMPLE_RATE * 0.05), dtype=np.int16)
    spike = np.ones(int(audio.SAMPLE_RATE * 0.01), dtype=np.int16) * 30000
    gap = np.zeros(int(audio.SAMPLE_RATE * 0.15), dtype=np.int16)
    tone = np.ones(int(audio.SAMPLE_RATE * 0.6), dtype=np.int16) * 15000
    quiet_trail = np.zeros(int(audio.SAMPLE_RATE * 0.2), dtype=np.int16)
    samples = np.concatenate([quiet_lead, spike, gap, tone, quiet_trail])

    bounds = extraction.propose_boundaries(samples, ["W", "IY"])

    # The spike is at ~50 ms, the actual speech starts at ~210 ms.
    # Boundary should be in the speech region, not at the spike.
    assert bounds[0] >= 180   # well past the spike
    # And ends before the trailing silence (speech ends at ~810 ms).
    assert bounds[-1] <= 840


def test_propose_boundaries_clips_low_level_noise_at_edges():
    """Background noise at start/end above the old 2% threshold but still
    well below speech level must be excluded from the loud region."""
    rng = np.random.default_rng(42)
    # 200 ms of low-level noise at ~5% of speech amplitude on each side.
    n_noise = int(audio.SAMPLE_RATE * 0.2)
    n_tone = int(audio.SAMPLE_RATE * 0.6)
    noise_lead = (rng.standard_normal(n_noise) * 1000).astype(np.int16)
    tone = np.ones(n_tone, dtype=np.int16) * 20000
    noise_trail = (rng.standard_normal(n_noise) * 1000).astype(np.int16)
    samples = np.concatenate([noise_lead, tone, noise_trail])

    bounds = extraction.propose_boundaries(samples, ["HH", "AH"])

    # The loud region should be approximately [200 ms, 800 ms] — the tone.
    # Allow ±40 ms slack for window quantisation (RMS_WINDOW_MS = 20).
    assert 160 <= bounds[0] <= 240
    assert 760 <= bounds[-1] <= 840
    # The full loud span should be most of the 600 ms tone, not the full 1000 ms.
    assert bounds[-1] - bounds[0] >= 500
    assert bounds[-1] - bounds[0] <= 700
