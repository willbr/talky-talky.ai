"""Boundary proposer: sentence, word and phoneme spans for a recording.

Pure numpy, no learned weights. Two steps:

1. `propose_sentence_spans` splits a long take into sentences using the
   text's sentence breaks and the longest pauses in the audio.
2. `propose` aligns one sentence with DTW against a reference synthesised
   from the inventory (`build_reference`), whose word and phoneme spans are
   known exactly, then maps those spans onto the recording. User-placed pins
   are hard anchors: DTW runs independently between them.

The function boundaries are deliberate: a future tinygrad/GPU proposer can
replace `propose` without touching storage or the UI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

import audio
import extraction
import store
import synth

log = logging.getLogger("align")

SR = audio.SAMPLE_RATE
N_FFT = 512
HOP = 220                         # ~10 ms
HOP_MS = HOP * 1000 / SR
N_MELS = 40
N_CEPS = 13
EDGE_PAD_MS = 100                 # silence around the reference / padding around the span
COMMA_PAUSE_MS = 150
MIN_PAUSE_MS = 150                # sentence-break pauses in the recording
MIN_SEG_MS = 5

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


# --- text ----------------------------------------------------------------------

def split_text_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text.strip())]
    return [p for p in parts if synth.tokenise(p)]


def _oov_ms(word: str) -> int:
    return max(150, 60 * len(word))


def _expected_ms(text: str) -> int:
    total = 0
    for tc in synth.phrase_context(text):
        if tc.phonemes:
            total += sum(extraction.PHONEME_DURATION_MS.get(p, extraction.DEFAULT_DURATION_MS)
                         for p in tc.phonemes)
        else:
            total += _oov_ms(tc.token)
        if tc.position == "comma":
            total += COMMA_PAUSE_MS
    return max(total, 1)


# --- step 1: sentences -------------------------------------------------------------

def _pauses_ms(samples: np.ndarray, lo_ms: int, hi_ms: int) -> list[tuple[int, int]]:
    """Quiet runs of at least MIN_PAUSE_MS inside [lo_ms, hi_ms]."""
    win = int(SR * extraction.RMS_WINDOW_MS / 1000)
    n = len(samples) // win
    if n < 3:
        return []
    rms = np.sqrt(np.mean(samples[: n * win].astype(np.float32).reshape(n, win) ** 2, axis=1))
    peak = float(rms.max())
    if peak == 0:
        return []
    floor = float(np.percentile(rms, extraction.NOISE_FLOOR_PERCENTILE))
    thr = min(max(floor * extraction.NOISE_FLOOR_MULTIPLIER,
                  peak * extraction.MIN_THRESHOLD_PEAK_RATIO),
              peak * extraction.MAX_THRESHOLD_PEAK_RATIO)
    quiet = rms <= thr
    min_run = max(1, int(MIN_PAUSE_MS / extraction.RMS_WINDOW_MS))
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            s, e = i * extraction.RMS_WINDOW_MS, j * extraction.RMS_WINDOW_MS
            if j - i >= min_run and s > lo_ms and e < hi_ms:
                out.append((s, e))
            i = j
        else:
            i += 1
    return out


def propose_sentence_spans(samples: np.ndarray, sentences: list[str]) -> list[tuple[int, int]]:
    lo, hi = extraction._loud_region_ms(samples, SR)
    if len(sentences) <= 1:
        return [(lo, hi)]
    weights = np.array([_expected_ms(s) for s in sentences], dtype=np.float64)
    expected = lo + np.cumsum(weights)[:-1] / weights.sum() * (hi - lo)
    pauses = _pauses_ms(samples, lo, hi)
    breaks: list[tuple[int, int]] = []
    after = lo
    for t in expected:
        cands = [p for p in pauses if p[0] >= after]
        if cands:
            best = min(cands, key=lambda p: abs((p[0] + p[1]) / 2 - t))
            pauses.remove(best)
        else:
            tt = int(max(t, after))
            best = (tt, tt)
        breaks.append(best)
        after = best[1]
    spans, start = [], lo
    for b in breaks:
        spans.append((start, max(start + MIN_SEG_MS, b[0])))
        start = b[1]
    spans.append((start, max(start + MIN_SEG_MS, hi)))
    return spans


# --- reference -------------------------------------------------------------------------

@dataclass
class Reference:
    samples: np.ndarray
    words: list[dict] = field(default_factory=list)     # {label, start_ms, end_ms, in_dict}
    phonemes: list[dict] = field(default_factory=list)  # {label, start_ms, end_ms, word_idx}


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


def build_reference(sentence_text: str, exclude=None) -> Reference:
    """Phonemes-only synthesis of the sentence with exactly known spans.

    Missing phoneme copies and out-of-dictionary words become silent
    placeholders of their expected length; commas get a short pause.
    """
    chunks: list[np.ndarray] = []
    ref = Reference(samples=np.array([], dtype=np.int16))
    t = 0.0

    def push(arr: np.ndarray) -> tuple[int, int]:
        nonlocal t
        start = t
        chunks.append(arr)
        t += len(arr) * 1000 / SR
        return int(round(start)), int(round(t))

    push(_silence(EDGE_PAD_MS))
    tcs = synth.phrase_context(sentence_text)
    for wi, tc in enumerate(tcs):
        w_start = int(round(t))
        if tc.phonemes:
            for ph, prev_ph, next_ph in synth.phoneme_neighbours(tc):
                unit = store.best_unit("phoneme", ph, prev=prev_ph, next=next_ph, exclude=exclude)
                if unit is not None:
                    samples, _ = audio.read_wav(store.unit_path(unit["id"]))
                    piece = audio.apply_fade(samples)
                else:
                    piece = _silence(extraction.PHONEME_DURATION_MS.get(ph, extraction.DEFAULT_DURATION_MS))
                s, e = push(piece)
                ref.phonemes.append({"label": ph, "start_ms": s, "end_ms": e, "word_idx": wi})
        else:
            push(_silence(_oov_ms(tc.token)))
        ref.words.append({"label": tc.token, "start_ms": w_start, "end_ms": int(round(t)),
                          "in_dict": tc.phonemes is not None})
        if tc.position != "mid" and wi < len(tcs) - 1:
            push(_silence(COMMA_PAUSE_MS))
    push(_silence(EDGE_PAD_MS))
    ref.samples = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    return ref


# --- features & DTW --------------------------------------------------------------------

_MEL_FB: np.ndarray | None = None
_DCT: np.ndarray | None = None


def _mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        mels = np.linspace(hz_to_mel(0), hz_to_mel(8000), N_MELS + 2)
        bins = np.floor((N_FFT + 1) * mel_to_hz(mels) / SR).astype(int)
        fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
        for m in range(1, N_MELS + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / max(1, c - l)
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / max(1, r - c)
        _MEL_FB = fb
    return _MEL_FB


def _dct() -> np.ndarray:
    global _DCT
    if _DCT is None:
        n = np.arange(N_MELS)
        k = np.arange(1, N_CEPS + 1)[:, None]
        _DCT = np.cos(np.pi / N_MELS * (n + 0.5) * k).astype(np.float32)
    return _DCT


def features(samples: np.ndarray) -> np.ndarray:
    """(frames, 14): MFCC c1..c13 + log energy, per-recording normalised.
    Frame i is centred on sample i*HOP."""
    x = samples.astype(np.float32) / 32768.0
    if len(x) == 0:
        return np.zeros((1, N_CEPS + 1), dtype=np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(N_FFT).astype(np.float32), axis=1)) ** 2
    logmel = np.log(power @ _mel_fb().T + 1e-10)
    ceps = logmel @ _dct().T
    energy = np.log(power.sum(axis=1) + 1e-10)[:, None]
    f = np.hstack([ceps, energy]).astype(np.float32)
    return (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-6)


def dtw_path(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimal (i, j) warp path from (0, 0) to (N-1, M-1), Euclidean cost,
    steps diag/up/left. Accumulated cost is filled one anti-diagonal at a
    time so each diagonal is a single vector operation."""
    n, m = len(a), len(b)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    cost = np.sqrt(np.maximum(a2 + b2 - 2 * a @ b.T, 0)).astype(np.float32)
    D = np.full((n, m), np.inf, dtype=np.float32)
    P = np.zeros((n, m), dtype=np.int8)       # 0 diag, 1 up (i-1), 2 left (j-1)
    D[0, 0] = cost[0, 0]
    for k in range(1, n + m - 1):
        i = np.arange(max(0, k - m + 1), min(n - 1, k) + 1)
        j = k - i
        im, jm = np.maximum(i - 1, 0), np.maximum(j - 1, 0)
        diag = np.where((i > 0) & (j > 0), D[im, jm], np.inf)
        up = np.where(i > 0, D[im, j], np.inf)
        left = np.where(j > 0, D[i, jm], np.inf)
        stack = np.stack([diag, up, left])
        choice = np.argmin(stack, axis=0)
        D[i, j] = cost[i, j] + stack[choice, np.arange(len(i))]
        P[i, j] = choice
    path = [(n - 1, m - 1)]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        c = P[i, j]
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return np.array(path[::-1], dtype=np.int64)


# --- step 2: words & phonemes -------------------------------------------------------------

def _ref_edge_ms(ref: Reference, pin: dict) -> int | None:
    segs = ref.words if pin.get("tier") == "word" else ref.phonemes
    idx = pin.get("idx")
    if not isinstance(idx, int) or not 0 <= idx < len(segs):
        return None
    return segs[idx]["start_ms" if pin.get("edge") == "start" else "end_ms"]


def propose(
    rec_samples: np.ndarray,
    span_ms: tuple[int, int],
    ref: Reference,
    pins: list[dict] | tuple = (),
) -> dict:
    """Map the reference's word/phoneme spans onto `rec_samples`.

    `span_ms` is the sentence's span in the recording. Pins are
    `{tier, idx, edge, ms}`; each pinned edge lands exactly on `ms`, the span
    widens to include every pin, and DTW runs separately between anchors.
    Returns `{words, phonemes, span_ms}` in absolute recording ms.
    """
    dur_ms = int(len(rec_samples) * 1000 / SR)
    pinned: dict[int, int] = {}          # ref edge ms -> recording ms
    for p in pins:
        r = _ref_edge_ms(ref, p)
        if r is not None and isinstance(p.get("ms"), (int, float)):
            pinned[r] = int(p["ms"])
    s, e = span_ms
    if pinned:
        s = min(s, min(pinned.values()) - 50)
        e = max(e, max(pinned.values()) + 50)
    s = max(0, int(s) - EDGE_PAD_MS)
    e = min(dur_ms, int(e) + EDGE_PAD_MS)

    seg = rec_samples[int(s * SR / 1000): int(e * SR / 1000)]
    fr, fq = features(seg), features(ref.samples)
    nr, nq = len(fr), len(fq)

    # Anchors in (ref_frame, rec_frame), monotone in both.
    anchors = [(0, 0)]
    for r_ms, p_ms in sorted(pinned.items()):
        rf = min(nq - 1, max(0, round(r_ms / HOP_MS)))
        cf = min(nr - 1, max(0, round((p_ms - s) / HOP_MS)))
        if rf > anchors[-1][0] and cf > anchors[-1][1]:
            anchors.append((rf, cf))
    if anchors[-1] != (nq - 1, nr - 1):
        anchors.append((nq - 1, nr - 1))

    parts = []
    for (r0, c0), (r1, c1) in zip(anchors, anchors[1:]):
        sub = dtw_path(fq[r0:r1 + 1], fr[c0:c1 + 1]) + (r0, c0)
        parts.append(sub if not parts else sub[1:])
    path = np.vstack(parts) if parts else np.array([[0, 0]])
    _, first = np.unique(path[:, 0], return_index=True)
    ref_to_rec = path[first, 1]

    def m(ref_ms: int) -> int:
        if ref_ms in pinned:
            return pinned[ref_ms]
        f = min(nq - 1, max(0, round(ref_ms / HOP_MS)))
        return int(round(s + ref_to_rec[f] * HOP_MS))

    out_ph = [{"idx": i, "label": p["label"], "start_ms": m(p["start_ms"]),
               "end_ms": m(p["end_ms"]), "parent_idx": p["word_idx"]}
              for i, p in enumerate(ref.phonemes)]
    out_w = [{"idx": i, "label": w["label"], "start_ms": m(w["start_ms"]),
              "end_ms": m(w["end_ms"]), "in_dict": w["in_dict"]}
             for i, w in enumerate(ref.words)]

    # Monotone, non-degenerate spans; words hug their phonemes.
    last = 0
    for p in out_ph:
        p["start_ms"] = max(p["start_ms"], last)
        p["end_ms"] = max(p["end_ms"], p["start_ms"] + MIN_SEG_MS)
        last = p["end_ms"]
    last = 0
    for w in out_w:
        mine = [p for p in out_ph if p["parent_idx"] == w["idx"]]
        if mine:
            w["start_ms"], w["end_ms"] = mine[0]["start_ms"], mine[-1]["end_ms"]
        w["start_ms"] = max(w["start_ms"], last)
        w["end_ms"] = max(w["end_ms"], w["start_ms"] + MIN_SEG_MS)
        last = w["end_ms"]
    span = [out_w[0]["start_ms"], out_w[-1]["end_ms"]] if out_w else [s, e]
    return {"words": out_w, "phonemes": out_ph, "span_ms": span}
