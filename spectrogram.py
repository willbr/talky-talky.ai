"""STFT log-magnitude spectrogram for visualisation.

Pure numpy. Computes a Hanning-windowed STFT, takes |log10(magnitude)|,
and quantises to uint8 so it can be base64'd into a JSON response and
drawn straight onto a canvas in the browser.
"""
from __future__ import annotations

import numpy as np

N_FFT = 512
HOP = 256


def compute(
    samples: np.ndarray,
    sample_rate: int = 22050,
    hop: int = HOP,
    value_range: tuple[float, float] | None = None,
) -> dict:
    """Return a uint8 log-magnitude spectrogram for the given int16 samples.

    Result dict:
      data:        bytes — uint8 row-major, shape (n_frames, n_freqs)
      n_frames:    int   — time bins
      n_freqs:     int   — frequency bins (= N_FFT // 2 + 1)
      hop_ms:      float — milliseconds between adjacent frames
      duration_ms: int   — total audio duration the spectrogram covers
      centre_ms:   float — offset from a frame's start to its window centre
      lo, hi:      float — log-magnitude range mapped to 0..255

    `hop` sets the time step (smaller = sharper when zoomed in). Pass
    `value_range` (another call's lo/hi) to keep brightness consistent.
    """
    duration_ms = int(len(samples) * 1000 / sample_rate)
    n_freqs = N_FFT // 2 + 1
    hop_ms = hop * 1000 / sample_rate
    centre_ms = (N_FFT / 2) * 1000 / sample_rate

    if len(samples) < N_FFT:
        return {
            "data": b"",
            "n_frames": 0,
            "n_freqs": n_freqs,
            "hop_ms": hop_ms,
            "duration_ms": duration_ms,
            "centre_ms": centre_ms,
            "lo": 0.0,
            "hi": 0.0,
        }

    x = samples.astype(np.float32) / 32768.0
    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::hop]
    spec = np.fft.rfft(frames * window, axis=1)
    log_mag = np.log10(np.abs(spec) + 1e-6).astype(np.float32)

    if value_range is not None:
        lo, hi = value_range
    else:
        lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi <= lo:
        norm = np.zeros_like(log_mag, dtype=np.uint8)
    else:
        norm = (np.clip((log_mag - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)

    return {
        "data": norm.tobytes(),
        "n_frames": int(norm.shape[0]),
        "n_freqs": int(norm.shape[1]),
        "hop_ms": hop_ms,
        "duration_ms": duration_ms,
        "centre_ms": centre_ms,
        "lo": lo,
        "hi": hi,
    }
