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
SILENCE_THRESHOLD_RATIO = 0.02  # fraction of peak below which we consider a sample silent
MAX_TRIM_SECONDS = 0.5
PEAK_TARGET = int(32767 * 10 ** (-3 / 20))  # -3 dBFS in int16 full scale (~23197)
FADE_SECONDS = 0.005  # 5 ms linear fade-in/fade-out
INTER_WORD_SILENCE_SECONDS = 0.030
MIN_LENGTH_SECONDS = 0.1


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    assert samples.dtype == np.int16, f"expected int16, got {samples.dtype}"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def read_wav(source) -> tuple[np.ndarray, int]:
    """Read a WAV file. `source` may be a path or a binary file-like object."""
    if hasattr(source, "read"):
        ctx = wave.open(source, "rb")
    else:
        ctx = wave.open(str(source), "rb")
    with ctx as r:
        rate = r.getframerate()
        n = r.getnframes()
        frames = r.readframes(n)
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, rate


def trim_silence(samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return samples

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return np.array([], dtype=np.int16)

    threshold = int(peak * SILENCE_THRESHOLD_RATIO)
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return np.array([], dtype=np.int16)

    first, last = int(loud[0]), int(loud[-1])
    max_trim = int(SAMPLE_RATE * MAX_TRIM_SECONDS)

    # Trim silence from edges, but don't trim more than MAX_TRIM_SECONDS
    start = min(first, max_trim)
    end = max(last + 1, len(samples) - max_trim)

    return samples[start:end]


def peak_normalise(samples: np.ndarray) -> np.ndarray:
    """Scale samples so the loudest sample sits at PEAK_TARGET. Silent input is returned unchanged."""
    if len(samples) == 0:
        return samples
    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return samples
    scale = PEAK_TARGET / peak
    return (samples.astype(np.float32) * scale).clip(-32768, 32767).astype(np.int16)


def apply_fade(samples: np.ndarray) -> np.ndarray:
    """Apply a 5 ms linear fade-in at the start and fade-out at the end.

    Returns the input unchanged if it is shorter than two fade windows.
    """
    fade_n = int(SAMPLE_RATE * FADE_SECONDS)
    if len(samples) < 2 * fade_n:
        return samples

    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    out = samples.astype(np.float32).copy()
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out.clip(-32768, 32767).astype(np.int16)


def concatenate(paths: list[Path]) -> bytes:
    """Read each WAV at `paths`, apply fade, append with inter-word silence, return WAV bytes."""
    import io as _io

    silence_n = int(SAMPLE_RATE * INTER_WORD_SILENCE_SECONDS)
    silence = np.zeros(silence_n, dtype=np.int16)

    chunks: list[np.ndarray] = []
    for i, p in enumerate(paths):
        samples, _ = read_wav(p)
        samples = apply_fade(samples)
        chunks.append(samples)
        if i < len(paths) - 1:
            chunks.append(silence)

    output = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(output.tobytes())
    return buf.getvalue()
