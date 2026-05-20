"""Audio I/O and DSP primitives for talky-talky.ai.

Uses stdlib `wave` for I/O and numpy for sample manipulation.
All recordings in the inventory are 16-bit mono PCM at 22050 Hz.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16-bit signed PCM)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    assert samples.dtype == np.int16, f"expected int16, got {samples.dtype}"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as r:
        rate = r.getframerate()
        n = r.getnframes()
        frames = r.readframes(n)
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, rate
