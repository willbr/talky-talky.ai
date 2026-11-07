"""Sentence synthesis: tokenise -> resolve -> concatenate."""
from __future__ import annotations

from dataclasses import dataclass

import audio
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


def synthesise(
    sentence: str,
    phonemes_only: bool = False,
) -> tuple[bytes | None, list[str], list[dict]]:
    """Synthesise a sentence by stitching word and/or phoneme copies.

    For each token (with its neighbours and phrase position as context):
      1. Walk the resolver chain. A hit becomes a single-unit group.
      2. Otherwise, if the token is itself an ARPABET symbol, skip it.
      3. Otherwise phonemise via CMUdict and pick the best copy of each
         phoneme given its in-word neighbours. Any missing phoneme → skip
         the whole word (all-or-nothing).

    `phonemes_only=True` drops the word resolver from the chain.

    Returns (wav_bytes_or_None, skipped_tokens, units). Each unit is
    {"token", "start_ms", "end_ms", "unit_id", "phonemes"}; `unit_id` is the
    word copy used (None when built from phonemes) and each phoneme span is
    {"symbol", "start_ms", "end_ms", "unit_id"}.
    """
    toks = phrase_context(sentence)
    if not toks:
        return None, [], []

    if phonemes_only:
        resolvers: list[resolver.Resolver] = [resolver.PhonemeResolver()]
    else:
        resolvers = resolver.RESOLVERS

    groups: list[list[dict]] = []
    rendered_tokens: list[str] = []
    word_units: list[dict | None] = []
    group_phonemes: list[list[str] | None] = []  # None ⇒ single word copy
    skipped: list[str] = []

    for tc in toks:
        token = tc.token
        ctx = resolver.Context(prev=tc.prev_word, next=tc.next_word, position=tc.position)

        direct = None
        for r in resolvers:
            direct = r.resolve(token, ctx)
            if direct is not None:
                break

        if direct is not None:
            groups.append([direct])
            rendered_tokens.append(token)
            if direct["kind"] == "phoneme":
                word_units.append(None)
                group_phonemes.append([direct["label"]])
            else:
                word_units.append(direct)
                group_phonemes.append(None)
            continue

        if phonemes.is_phoneme(token):
            skipped.append(token)
            continue

        phs = tc.phonemes
        if phs is None:
            skipped.append(token)
            continue

        picked: list[dict] = []
        for ph, prev_ph, next_ph in phoneme_neighbours(tc):
            unit = store.best_unit("phoneme", ph, prev=prev_ph, next=next_ph)
            if unit is None:
                break
            picked.append(unit)

        if len(picked) != len(phs):
            skipped.append(token)
            continue
        groups.append(picked)
        rendered_tokens.append(token)
        word_units.append(None)
        group_phonemes.append(list(phs))

    if not groups:
        return None, skipped, []

    path_groups = [[store.unit_path(u["id"]) for u in g] for g in groups]
    units = _compute_units(path_groups, groups, rendered_tokens, word_units, group_phonemes)
    wav_bytes = audio.concatenate(path_groups)
    return wav_bytes, skipped, units


def _compute_units(
    path_groups: list[list],
    unit_groups: list[list[dict]],
    rendered_tokens: list[str],
    word_units: list[dict | None],
    group_phonemes: list[list[str] | None],
) -> list[dict]:
    """Assign each rendered token (and each phoneme inside it) a
    [start_ms, end_ms] span in the stitched audio, mirroring
    `audio.concatenate`'s inter-group silence. Durations come from the WAV
    files themselves, not cached metadata."""
    inter_ms = audio.INTER_WORD_SILENCE_SECONDS * 1000
    cumulative = 0.0
    units: list[dict] = []
    n = len(path_groups)
    for i in range(n):
        unit_start = cumulative
        phs = group_phonemes[i]
        spans: list[dict] = []
        for path, unit, sym in zip(path_groups[i], unit_groups[i], phs or [None] * len(path_groups[i])):
            samples, _ = audio.read_wav(path)
            d_ms = len(samples) * 1000 / audio.SAMPLE_RATE
            if phs is not None:
                spans.append({
                    "symbol": sym,
                    "start_ms": int(round(cumulative)),
                    "end_ms": int(round(cumulative + d_ms)),
                    "unit_id": unit["id"],
                })
            cumulative += d_ms
        units.append({
            "token": rendered_tokens[i],
            "start_ms": int(round(unit_start)),
            "end_ms": int(round(cumulative)),
            "unit_id": word_units[i]["id"] if word_units[i] else None,
            "source": word_units[i]["source"] if word_units[i] else None,
            "source_recording_id": word_units[i]["source_recording_id"] if word_units[i] else None,
            "source_sentence_idx": word_units[i]["source_sentence_idx"] if word_units[i] else None,
            "phonemes": spans,
        })
        if i < n - 1:
            cumulative += inter_ms
    return units
