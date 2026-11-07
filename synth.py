"""Sentence synthesis: tokenise -> resolve -> concatenate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import audio
import diphones
import inventory
import phonemes
import resolver
import store

_FINAL_PUNCT = set(".!?")
_COMMA_PUNCT = set(",;:")


def tokenise(sentence: str) -> list[str]:
    """Split sentence into normalised tokens. Empty tokens are dropped."""
    return [t for t, _ in tokenise_with_positions(sentence)]


def _trailing(raw: str) -> str:
    """Characters after the last letter/digit of a raw token."""
    i = len(raw)
    while i > 0 and not raw[i - 1].isalnum():
        i -= 1
    return raw[i:]


def _position(punct: str) -> str:
    if _FINAL_PUNCT & set(punct):
        return "final"
    if _COMMA_PUNCT & set(punct):
        return "comma"
    return "mid"


def tokenise_with_positions(sentence: str) -> list[tuple[str, str]]:
    """Normalised tokens paired with their phrase position.

    A token's position comes from the punctuation after it, up to the next
    kept token: `. ! ?` → "final", else `, ; :` → "comma", else "mid". The
    last token is always "final". Punctuation-only raw tokens ("wait , as")
    attach to the preceding token.
    """
    kept: list[list[str]] = []   # [token, trailing punctuation]
    for raw in sentence.split():
        norm = inventory.normalise(raw)
        if norm:
            kept.append([norm, _trailing(raw)])
        elif kept:
            kept[-1][1] += raw
    out = [(tok, _position(punct)) for tok, punct in kept]
    if out:
        out[-1] = (out[-1][0], "final")
    return out


@dataclass(frozen=True)
class TokenCtx:
    """A token plus the context synth and harvest both use to pick copies.

    Context never crosses punctuation: prev/next are None across `, ; :`
    and `. ! ?` (a pause separates the sounds), and at the text's ends.
    """
    token: str
    position: str
    prev_word: str | None
    next_word: str | None
    phonemes: list[str] | None      # CMUdict pronunciation, None if not in dict
    prev_phoneme: str | None        # neighbour of the first phoneme (cross-word)
    next_phoneme: str | None        # neighbour of the last phoneme (cross-word)


def phrase_context(text: str) -> list[TokenCtx]:
    toks = tokenise_with_positions(text)
    prons = [phonemes.word_to_phonemes(t) for t, _ in toks]
    out: list[TokenCtx] = []
    for i, (tok, pos) in enumerate(toks):
        joined_prev = i > 0 and toks[i - 1][1] == "mid"
        joined_next = i < len(toks) - 1 and pos == "mid"
        prev_pron = prons[i - 1] if joined_prev else None
        next_pron = prons[i + 1] if joined_next else None
        out.append(TokenCtx(
            token=tok,
            position=pos,
            prev_word=toks[i - 1][0] if joined_prev else None,
            next_word=toks[i + 1][0] if joined_next else None,
            phonemes=prons[i],
            prev_phoneme=prev_pron[-1] if prev_pron else None,
            next_phoneme=next_pron[0] if next_pron else None,
        ))
    return out


def phoneme_neighbours(tc: TokenCtx) -> list[tuple[str, str | None, str | None]]:
    """(symbol, prev, next) for each phoneme of `tc`, crossing word edges."""
    phs = tc.phonemes or []
    return [
        (ph,
         phs[j - 1] if j > 0 else tc.prev_phoneme,
         phs[j + 1] if j < len(phs) - 1 else tc.next_phoneme)
        for j, ph in enumerate(phs)
    ]


COMMA_PAUSE_MS = 150
FINAL_PAUSE_MS = 300
JOIN_OVERLAP_MS = 15      # crossfade between diphones inside a run
JOIN_SEARCH_MS = 8        # how far the incoming piece may slide to line up in phase
HALF_JOIN_MS = 5          # crossfade between two phoneme-copy halves (fallback)


def _ms_n(ms: float) -> int:
    return int(round(ms * audio.SAMPLE_RATE / 1000))


class _Missing(Exception):
    """A run slot could be filled neither by a diphone nor by phoneme halves."""

    def __init__(self, word_idx: int):
        super().__init__(word_idx)
        self.word_idx = word_idx


def _join(a: np.ndarray, b: np.ndarray, overlap_n: int, search_n: int) -> tuple[np.ndarray, int]:
    """Crossfade `b` onto the end of `a`.

    Before fading, `b` may slide up to `search_n` samples to the offset whose
    start best correlates with `a`'s tail, so periodic sounds meet in phase.
    Returns (joined, index in the result where b[0] would sit; samples of b
    before the slide offset are dropped).
    """
    n = min(overlap_n, len(a), max(0, len(b) - search_n) // 2)
    if n <= 0:
        return np.concatenate([a, b]), len(a)
    search = max(0, min(search_n, len(b) - 2 * n))
    o = 0
    if search > 0:
        tail = a[-n:]
        win = np.lib.stride_tricks.sliding_window_view(b[:n + search], n)
        den = np.sqrt((win * win).sum(axis=1) * float(tail @ tail)) + 1e-9
        o = int(np.argmax((win @ tail) / den))
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    mixed = a[-n:] * (1 - ramp) + b[o:o + n] * ramp
    return np.concatenate([a[:-n], mixed, b[o + n:]]), len(a) - n - o


def _fade(x: np.ndarray) -> np.ndarray:
    n = min(_ms_n(audio.FADE_SECONDS * 1000), len(x) // 2)
    if n <= 0:
        return x
    x = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def synthesise(
    sentence: str,
    phonemes_only: bool = False,
) -> tuple[bytes | None, list[str], list[dict], dict]:
    """Synthesise a sentence from word copies and diphones.

    Each token (with its neighbours and phrase position as context) becomes:
      - a word copy, or a phoneme copy if the token is an ARPABET symbol,
        via the resolver chain (`phonemes_only` drops the word resolver);
      - else its CMUdict pronunciation joins the current *run*. A run ends
        at a copy, a skipped token, or punctuation, and is rendered as
        `SIL p1 … pn SIL`, one diphone per adjacent pair, crossfaded
        mid-phoneme. A missing diphone is stood in for by halves of the two
        phoneme copies; if a phoneme has no copy either, its word is
        skipped and the run splits around it.

    Comma tokens are followed by COMMA_PAUSE_MS of silence, final tokens
    (except the last) by FINAL_PAUSE_MS.

    Returns (wav_bytes_or_None, skipped_tokens, units, diphone_stats). Each
    unit is {"token", "start_ms", "end_ms", "unit_id", "source",
    "source_recording_id", "source_sentence_idx", "phonemes"}; `unit_id` is
    the word copy used (None otherwise) and each phoneme span is
    {"symbol", "start_ms", "end_ms"}. `diphone_stats` is {"real", "fallback"}.
    """
    toks = phrase_context(sentence)
    stats = {"real": 0, "fallback": 0}
    if not toks:
        return None, [], [], stats

    resolvers: list[resolver.Resolver] = (
        [resolver.PhonemeResolver()] if phonemes_only else resolver.RESOLVERS)

    items: list[tuple] = []           # ("copy", i, unit) | ("run", [i…]) | ("pause", ms)
    skipped_idx: set[int] = set()
    run: list[int] = []

    def flush() -> None:
        if run:
            items.append(("run", run.copy()))
            run.clear()

    for i, tc in enumerate(toks):
        ctx = resolver.Context(prev=tc.prev_word, next=tc.next_word, position=tc.position)
        direct = None
        for r in resolvers:
            direct = r.resolve(tc.token, ctx)
            if direct is not None:
                break
        if direct is not None:
            flush()
            items.append(("copy", i, direct))
        elif phonemes.is_phoneme(tc.token) or not tc.phonemes:
            flush()
            skipped_idx.add(i)
        else:
            run.append(i)
        if tc.position != "mid":
            flush()
            if i < len(toks) - 1:
                items.append(("pause", COMMA_PAUSE_MS if tc.position == "comma" else FINAL_PAUSE_MS))
    flush()

    cache: dict[str, np.ndarray] = {}

    def read(uid: str) -> np.ndarray:
        if uid not in cache:
            samples, _ = audio.read_wav(store.unit_path(uid))
            cache[uid] = samples.astype(np.float32)
        return cache[uid]

    # (samples, [(token_idx, word_unit|None, [(symbol, start_n, end_n)], start_n, end_n)])
    pieces: list[tuple[np.ndarray, list]] = []

    def render_run(idxs: list[int]) -> tuple[np.ndarray, list]:
        phones = [(wi, ph) for wi in idxs for ph in toks[wi].phonemes]
        syms = [ph for _, ph in phones]
        pad = _ms_n(diphones.PAD_MS)
        copies: dict[int, dict] = {}

        def phone_copy(j: int) -> np.ndarray:
            if j not in copies:
                u = store.best_unit("phoneme", syms[j],
                                    prev=syms[j - 1] if j > 0 else None,
                                    next=syms[j + 1] if j + 1 < len(syms) else None)
                if u is None:
                    raise _Missing(phones[j][0])
                copies[j] = u
            return read(copies[j]["id"])

        out = np.zeros(0, dtype=np.float32)
        bounds: list[int] = []        # bounds[j] = start of phone j; bounds[-1] = end
        overlap, search = _ms_n(JOIN_OVERLAP_MS), _ms_n(JOIN_SEARCH_MS)
        for k, (lab, prev, nxt) in enumerate(diphones.sequence_labels(syms)):
            d = store.best_unit("diphone", lab, prev=prev, next=nxt)
            if d is not None:
                piece = read(d["id"])
                b_in = _ms_n(d["boundary_ms"] or 0)
            else:
                right = np.zeros(0, dtype=np.float32)
                left = np.zeros(0, dtype=np.float32)
                if k > 0:
                    x = phone_copy(k - 1)
                    right = x[max(0, len(x) // 2 - pad):]
                if k < len(syms):
                    x = phone_copy(k)
                    left = x[:len(x) // 2 + pad]
                piece, b_in = _join(right, left, _ms_n(HALF_JOIN_MS), 0)
            if k == 0:
                out, at = piece, 0
            else:
                out, at = _join(out, piece, overlap, search)
            bounds.append(max(bounds[-1] if bounds else 0, at + b_in))
            stats["real" if d is not None else "fallback"] += 1
        bounds = [min(b, len(out)) for b in bounds]

        spans = []
        j = 0
        for wi in idxs:
            n = len(toks[wi].phonemes)
            phs = [(syms[j + q], bounds[j + q], bounds[j + q + 1]) for q in range(n)]
            spans.append((wi, None, phs, bounds[j], bounds[j + n]))
            j += n
        return _fade(out), spans

    def render_run_safe(idxs: list[int]) -> None:
        # Diphone/fallback counts of a run that fails are discarded with it.
        before = dict(stats)
        try:
            pieces.append(render_run(idxs))
        except _Missing as m:
            stats.update(before)
            skipped_idx.add(m.word_idx)
            cut = idxs.index(m.word_idx)
            for part in (idxs[:cut], idxs[cut + 1:]):
                if part:
                    render_run_safe(part)

    for item in items:
        if item[0] == "copy":
            _, i, unit = item
            x = audio.apply_fade(read(unit["id"]).astype(np.int16)).astype(np.float32)
            word = unit if unit["kind"] == "word" else None
            phs = [] if word else [(unit["label"], 0, len(x))]
            pieces.append((x, [(i, word, phs, 0, len(x))]))
        elif item[0] == "run":
            render_run_safe(item[1])
        else:
            pieces.append((np.zeros(_ms_n(item[1]), dtype=np.float32), []))

    skipped = [toks[i].token for i in sorted(skipped_idx)]
    # Pauses only matter between speech.
    while pieces and not pieces[-1][1]:
        pieces.pop()
    while pieces and not pieces[0][1]:
        pieces.pop(0)
    if not pieces:
        return None, skipped, [], stats

    def ms(n: int) -> int:
        return int(round(n * 1000 / audio.SAMPLE_RATE))

    units: list[dict] = []
    offset = 0
    for x, spans in pieces:
        for wi, word, phs, s, e in spans:
            units.append({
                "token": toks[wi].token,
                "start_ms": ms(offset + s),
                "end_ms": ms(offset + e),
                "unit_id": word["id"] if word else None,
                "source": word["source"] if word else None,
                "source_recording_id": word["source_recording_id"] if word else None,
                "source_sentence_idx": word["source_sentence_idx"] if word else None,
                "phonemes": [{"symbol": sym, "start_ms": ms(offset + ps), "end_ms": ms(offset + pe)}
                             for sym, ps, pe in phs],
            })
        offset += len(x)
    samples = np.concatenate([x for x, _ in pieces]).clip(-32768, 32767).astype(np.int16)
    return audio.encode_wav(samples), skipped, units, stats
