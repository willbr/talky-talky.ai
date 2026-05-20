import numpy as np
import audio


def test_write_then_read_preserves_samples(tmp_path):
    samples = np.array([0, 1000, -1000, 32000, -32000, 0], dtype=np.int16)
    path = tmp_path / "round_trip.wav"

    audio.write_wav(path, samples)
    loaded, rate = audio.read_wav(path)

    assert rate == audio.SAMPLE_RATE
    assert loaded.dtype == np.int16
    np.testing.assert_array_equal(loaded, samples)


def test_constants():
    assert audio.SAMPLE_RATE == 22050
    assert audio.CHANNELS == 1
    assert audio.SAMPLE_WIDTH == 2


def test_trim_silence_removes_padding():
    silence = np.zeros(1000, dtype=np.int16)
    loud = (np.ones(500, dtype=np.int16) * 10000)
    signal = np.concatenate([silence, loud, silence])

    trimmed = audio.trim_silence(signal)

    assert len(trimmed) == 500
    np.testing.assert_array_equal(trimmed, loud)


def test_trim_silence_all_silent_returns_empty():
    signal = np.zeros(5000, dtype=np.int16)
    trimmed = audio.trim_silence(signal)
    assert len(trimmed) == 0


def test_trim_silence_caps_at_max_trim():
    # 2 seconds of silence followed by 100 ms of tone — trim should cap at 500 ms
    silence_samples = audio.SAMPLE_RATE * 2
    tone_samples = int(audio.SAMPLE_RATE * 0.1)
    signal = np.concatenate([
        np.zeros(silence_samples, dtype=np.int16),
        np.ones(tone_samples, dtype=np.int16) * 10000,
    ])

    trimmed = audio.trim_silence(signal)

    max_trim = int(audio.SAMPLE_RATE * 0.5)
    expected_min_len = (silence_samples - max_trim) + tone_samples
    assert len(trimmed) >= expected_min_len


def test_peak_normalise_scales_to_target():
    samples = np.array([0, 1000, -1000, 500], dtype=np.int16)
    normalised = audio.peak_normalise(samples)
    assert int(np.max(np.abs(normalised))) == audio.PEAK_TARGET


def test_peak_normalise_silent_input_unchanged():
    silent = np.zeros(100, dtype=np.int16)
    out = audio.peak_normalise(silent)
    np.testing.assert_array_equal(out, silent)


def test_peak_target_constant():
    # -3 dBFS relative to int16 full scale
    expected = int(32767 * 10 ** (-3 / 20))
    assert audio.PEAK_TARGET == expected


def test_apply_fade_ramps_start_and_end():
    samples = np.ones(2000, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)

    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    # First sample of fade-in is ~0, ramps up
    assert abs(int(faded[0])) < 100
    assert int(faded[fade_n // 2]) < int(faded[fade_n - 1])
    # Last sample of fade-out is ~0, ramps down
    assert abs(int(faded[-1])) < 100
    assert int(faded[-fade_n // 2]) < int(faded[-fade_n])


def test_apply_fade_leaves_middle_unchanged():
    samples = np.ones(2000, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)

    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    middle = faded[fade_n:-fade_n]
    np.testing.assert_array_equal(middle, samples[fade_n:-fade_n])


def test_apply_fade_short_signal_returns_unchanged():
    # Signal shorter than 2 * fade window — fade can't be applied safely
    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    samples = np.ones(fade_n, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)
    np.testing.assert_array_equal(faded, samples)
