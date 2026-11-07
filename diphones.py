"""Diphones: units that run from the middle of one phoneme to the middle of
the next, so joins land in steady sounds and transitions stay whole.

Labels are `A-B`; `SIL` stands for a pause or the edge of the text. A
diphone's `prev`/`next` context is the phoneme before A / after B (None at
a pause), mirroring how phoneme copies carry their neighbours.

  - `cuts`: where to slice a saved splitter sentence (harvest).
  - `phrases` / `in_text`: the diphone sequence a text needs (synth, coverage).
  - `pick_for_coverage`: the book passage that adds the most missing diphones.
"""
from __future__ import annotations

from collections import Counter

SIL = "SIL"
PAD_MS = 12            # overlap material kept beyond each midpoint
MIN_MS = 20            # shorter cuts are not harvested
COVERAGE_CANDIDATES = 200
COVERAGE_MIN_CHARS = 150
COVERAGE_MAX_CHARS = 500   # longer passages are tiring to read in one take


def label(a: str, b: str) -> str:
    return f"{a}-{b}"


def _ctx(sym: str | None) -> str | None:
    return None if sym in (None, SIL) else sym


def phrases(tcs) -> list[list[tuple[int, str]]]:
    """Split `phrase_context` tokens into phrases of (word_idx, phoneme).

    A phrase ends after a comma/final token and around any word with no
    pronunciation (it has no phoneme segments, so nothing joins across it).
    """
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for wi, tc in enumerate(tcs):
        if not tc.phonemes:
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.extend((wi, ph) for ph in tc.phonemes)
        if tc.position != "mid":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def sequence_labels(phones: list[str]) -> list[tuple[str, str | None, str | None]]:
    """(label, prev, next) for every slot of `SIL p1 … pn SIL`."""
    seq = [SIL, *phones, SIL]
    return [
        (label(seq[i], seq[i + 1]),
         _ctx(seq[i - 1]) if i > 0 else None,
         _ctx(seq[i + 2]) if i + 2 < len(seq) else None)
        for i in range(len(seq) - 1)
    ]


def in_text(text: str) -> list[str]:
    """Every diphone label the text needs, in order (with repeats)."""
    import synth
    return [lab for ph in phrases(synth.phrase_context(text))
            for lab, _, _ in sequence_labels([p for _, p in ph])]


def cuts(tcs, phoneme_segments: list[dict]) -> list[dict]:
    """Diphone slices for one splitter sentence.

    `phoneme_segments` are the sentence's phoneme rows in order, each with
    `parent_idx` (word index), `start_ms`, `end_ms`. Returns dicts
    `{label, start_ms, end_ms, boundary_ms, prev, next, source_word}` where
    `boundary_ms` is the A|B boundary relative to `start_ms`. Unclamped:
    the caller clamps to the recording.
    """
    segs = iter(phoneme_segments)
    seg_of: list[dict] = []
    words: list[int] = []
    phone_phrases = phrases(tcs)
    # phoneme_segments are in the same order as the phrases' phonemes, minus
    # nothing: out-of-dictionary words contribute no phonemes to either.
    for ph in phone_phrases:
        for wi, _ in ph:
            seg_of.append(next(segs))
            words.append(wi)

    out: list[dict] = []
    k = 0
    for ph in phone_phrases:
        n = len(ph)
        segs_p = seg_of[k:k + n]
        words_p = words[k:k + n]
        k += n
        mids = [(s["start_ms"] + s["end_ms"]) / 2 for s in segs_p]
        for i, (lab, prev, nxt) in enumerate(sequence_labels([p for _, p in ph])):
            a, b = i - 1, i          # phrase phoneme indices; -1 / n are SIL
            if a < 0:
                start, end = segs_p[b]["start_ms"], mids[b] + PAD_MS
                boundary = start
                word = tcs[words_p[b]].token
            elif b >= n:
                start, end = mids[a] - PAD_MS, segs_p[a]["end_ms"]
                boundary = end
                word = tcs[words_p[a]].token
            else:
                start, end = mids[a] - PAD_MS, mids[b] + PAD_MS
                boundary = (segs_p[a]["end_ms"] + segs_p[b]["start_ms"]) / 2
                wa, wb = tcs[words_p[a]].token, tcs[words_p[b]].token
                word = wa if words_p[a] == words_p[b] else f"{wa} {wb}"
            out.append({"label": lab, "start_ms": start, "end_ms": end,
                        "boundary_ms": boundary - start, "prev": prev, "next": nxt,
                        "source_word": word})
    return out


# --- coverage -------------------------------------------------------------------------

def pick_for_coverage(book: str | None, have: set[str]) -> dict | None:
    """Of COVERAGE_CANDIDATES random passages, the one whose missing diphones
    weigh most. Weights are each diphone's frequency across the candidate
    pool, so common gaps are filled first. Returns
    `{book, passage, new_diphones, have}` or None."""
    import books

    pool: list[tuple[str, str, Counter]] = []
    seen: set[str] = set()
    for _ in range(COVERAGE_CANDIDATES):
        got = books.sample_passage(book)
        if got is None:
            break
        if got[1] in seen:
            continue
        seen.add(got[1])
        pool.append((got[0], got[1], Counter(in_text(got[1]))))
    if not pool:
        return None
    freq: Counter = Counter()
    for _, _, c in pool:
        freq.update(c.keys())

    # Gain over sqrt(length): longer passages must earn their extra reading,
    # but a one-liner can't win on ratio alone.
    def score(item) -> float:
        missing = [d for d in item[2] if d not in have]
        return sum(freq[d] for d in missing) / max(1, sum(item[2].values())) ** 0.5

    short = [c for c in pool if COVERAGE_MIN_CHARS <= len(c[1]) <= COVERAGE_MAX_CHARS] or pool
    b, passage, counts = max(short, key=score)
    return {"book": b, "passage": passage,
            "new_diphones": sum(1 for d in counts if d not in have), "have": len(have)}
