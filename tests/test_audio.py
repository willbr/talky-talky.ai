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
