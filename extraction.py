"""Duration-weighted phoneme boundary proposal.

Pure math; no I/O. Operates on int16 samples and a phoneme sequence,
returns N+1 boundary times in milliseconds for N phonemes.

The first and last boundaries are clamped to the first/last LOUD sample
(peak * 2%), so silent edges are excluded by default. Word-internal silences
are not detected; users fix them by dragging in the editor.
"""
from __future__ import annotations

import logging

import numpy as np

import phonemes

log = logging.getLogger("extraction")

SAMPLE_RATE = 22050
SILENCE_THRESHOLD_RATIO = 0.02
DEFAULT_DURATION_MS = 100

PHONEME_DURATION_MS: dict[str, int] = {
    # vowels (~150 ms)
    "AA": 150, "AE": 150, "AH": 150, "AO": 150, "AW": 150, "AY": 150,
    "EH": 150, "ER": 150, "EY": 150, "IH": 150, "IY": 150, "OW": 150,
    "OY": 150, "UH": 150, "UW": 150,
    # stops (~80 ms)
    "B": 80, "D": 80, "G": 80, "K": 80, "P": 80, "T": 80,
    # nasals (~90 ms)
    "M": 90, "N": 90, "NG": 90,
    # fricatives (~120 ms)
    "DH": 120, "F": 120, "HH": 120, "S": 120, "SH": 120,
    "TH": 120, "V": 120, "Z": 120, "ZH": 120,
    # liquids (~90 ms)
    "L": 90, "R": 90,
    # glides (~80 ms)
    "W": 80, "Y": 80,
    # affricates (~130 ms)
    "CH": 130, "JH": 130,
}


def _loud_region_ms(samples: np.ndarray, sample_rate: int) -> tuple[int, int]:
    """Return (start_ms, end_ms) covering the loud region of `samples`."""
    duration_ms = int(len(samples) * 1000 / sample_rate)
    if len(samples) == 0:
        return 0, 0

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        log.warning("propose_boundaries: all samples silent; using full audio range")
        return 0, duration_ms

    threshold = int(peak * SILENCE_THRESHOLD_RATIO)
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        log.warning("propose_boundaries: no samples above threshold; using full audio range")
        return 0, duration_ms

    first, last = int(loud[0]), int(loud[-1])
    start_ms = int(first * 1000 / sample_rate)
    end_ms = int((last + 1) * 1000 / sample_rate)
    return start_ms, end_ms


def propose_boundaries(
    samples: np.ndarray,
    phoneme_seq: list[str],
    sample_rate: int = SAMPLE_RATE,
) -> list[int]:
    """Return N+1 boundary times in milliseconds for N phonemes."""
    n = len(phoneme_seq)
    if n == 0:
        return [0]

    start_ms, end_ms = _loud_region_ms(samples, sample_rate)
    span = max(0, end_ms - start_ms)

    priors: list[int] = []
    for p in phoneme_seq:
        sym = p.upper()
        if sym in PHONEME_DURATION_MS:
            priors.append(PHONEME_DURATION_MS[sym])
        else:
            log.warning("propose_boundaries: unknown phoneme %r; using default %d ms",
                        p, DEFAULT_DURATION_MS)
            priors.append(DEFAULT_DURATION_MS)

    total_prior = sum(priors)
    if total_prior == 0:
        return [start_ms] + [start_ms] * (n - 1) + [end_ms]

    bounds = [start_ms]
    cumulative = 0
    for prior in priors[:-1]:
        cumulative += prior
        bounds.append(start_ms + int(cumulative * span / total_prior))
    bounds.append(end_ms)
    return bounds
