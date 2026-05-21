import io

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


def test_concatenate_combines_two_wavs(tmp_path):
    a_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000  # 1 sec
    b_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 5000   # 1 sec
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    audio.write_wav(a_path, a_samples)
    audio.write_wav(b_path, b_samples)

    wav_bytes = audio.concatenate([a_path, b_path])

    # Load result back via stdlib wave to validate header + length
    loaded, rate = audio.read_wav(io.BytesIO(wav_bytes))
    assert rate == audio.SAMPLE_RATE
    expected_silence = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    expected_len = len(a_samples) + expected_silence + len(b_samples)
    assert len(loaded) == expected_len


def test_concatenate_single_path_no_silence_appended(tmp_path):
    samples = np.ones(1000, dtype=np.int16) * 10000
    path = tmp_path / "only.wav"
    audio.write_wav(path, samples)

    wav_bytes = audio.concatenate([path])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == len(samples)


def test_concatenate_empty_list_returns_empty_wav():
    wav_bytes = audio.concatenate([])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == 0


def test_concatenate_grouped_silence_between_groups_only(tmp_path):
    """[[a, b], [c]] -> a + b + 30ms + c (no silence between a and b)."""
    samples_a = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    samples_b = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 8000
    samples_c = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 6000
    a = tmp_path / "a.wav"; audio.write_wav(a, samples_a)
    b = tmp_path / "b.wav"; audio.write_wav(b, samples_b)
    c = tmp_path / "c.wav"; audio.write_wav(c, samples_c)

    wav_bytes = audio.concatenate([[a, b], [c]])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))

    inter = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    expected_len = len(samples_a) + len(samples_b) + inter + len(samples_c)
    assert len(loaded) == expected_len


def test_concatenate_grouped_drops_empty_inner_groups(tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    p = tmp_path / "a.wav"; audio.write_wav(p, samples)

    wav_bytes = audio.concatenate([[], [p], []])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == len(samples)


def test_concatenate_grouped_empty_outer_list_returns_empty_wav():
    wav_bytes = audio.concatenate([[], []])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == 0


def test_concatenate_flat_list_still_works(tmp_path):
    """Backward compatibility: flat list[Path] still inserts inter-word silence."""
    samples_a = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    samples_b = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 8000
    a = tmp_path / "a.wav"; audio.write_wav(a, samples_a)
    b = tmp_path / "b.wav"; audio.write_wav(b, samples_b)

    wav_bytes = audio.concatenate([a, b])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))

    inter = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    assert len(loaded) == len(samples_a) + inter + len(samples_b)


def test_slice_samples_basic():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)  # exactly 1 second
    out = audio.slice_samples(s, 100, 200)
    expected_len = int(audio.SAMPLE_RATE * 0.1)  # 100 ms worth
    assert len(out) == expected_len


def test_slice_samples_clamps_out_of_range():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)
    out = audio.slice_samples(s, -50, 1500)
    assert len(out) == len(s)


def test_slice_samples_inverted_range_returns_empty():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)
    out = audio.slice_samples(s, 500, 200)
    assert len(out) == 0
