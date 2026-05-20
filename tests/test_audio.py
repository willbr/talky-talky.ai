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
