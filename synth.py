"""Sentence synthesis: tokenise -> resolve -> concatenate."""
from __future__ import annotations

import audio
import inventory
import phonemes
import resolver


def tokenise(sentence: str) -> list[str]:
    """Split sentence into normalised tokens. Empty tokens are dropped."""
    raw_tokens = sentence.split()
    tokens = [inventory.normalise(t) for t in raw_tokens]
    return [t for t in tokens if t]


def synthesise(
    sentence: str,
    phonemes_only: bool = False,
) -> tuple[bytes | None, list[str], list[dict]]:
    """Synthesise a sentence by stitching word recordings and/or phoneme recordings.

    For each token in the sentence:
      1. Walk the resolver chain. If a resolver returns a Path, use it as a
         single-element group.
      2. Otherwise, if the token is itself an ARPABET symbol that wasn't
         resolved (no phoneme file recorded), skip the token.
      3. Otherwise, look the token up in CMUdict. If absent → skip. If
         present, try the resolver chain on each phoneme. If every phoneme
         resolves, the word becomes a group of phoneme paths. If any phoneme
         is missing → skip the whole word (all-or-nothing).

    When `phonemes_only=True`, the word-recording resolver is dropped from
    the chain — tokens are always either ARPABET symbols or phonemised via
    CMUdict, never matched against `inventory/<word>.wav`.

    Returns (wav_bytes_or_None, skipped_tokens, units) where `units` is a
    list of {"token", "start_ms", "end_ms"} for each rendered group, in
    output order. Useful for annotating the synthesised audio.
    """
    tokens = tokenise(sentence)
    if not tokens:
        return None, [], []

    if phonemes_only:
        resolvers: list[resolver.Resolver] = [resolver.PhonemeResolver()]
    else:
        resolvers = resolver.RESOLVERS

    groups: list[list] = []
    rendered_tokens: list[str] = []
    group_phonemes: list[list[str] | None] = []  # parallel to groups; None ⇒ single word recording
    skipped: list[str] = []

    for token in tokens:
        # Pass 1: try the chain on the raw token.
        direct = None
        for r in resolvers:
            direct = r.resolve(token)
            if direct is not None:
                break

        if direct is not None:
            groups.append([direct])
            rendered_tokens.append(token)
            # Was the direct hit a word or a phoneme symbol? If it was a
            # phoneme symbol resolved against the phoneme inventory we still
            # know which phoneme it is — record it for annotation.
            group_phonemes.append([token.upper()] if phonemes.is_phoneme(token) else None)
            continue

        # If the token itself is a phoneme symbol, the chain already had its chance.
        if phonemes.is_phoneme(token):
            skipped.append(token)
            continue

        # Pass 2: phonemise via CMUdict, then resolve each phoneme.
        phs = phonemes.word_to_phonemes(token)
        if phs is None:
            skipped.append(token)
            continue

        phoneme_paths = []
        all_resolved = True
        for ph in phs:
            ph_path = None
            for r in resolvers:
                ph_path = r.resolve(ph)
                if ph_path is not None:
                    break
            if ph_path is None:
                all_resolved = False
                break
            phoneme_paths.append(ph_path)

        if not all_resolved:
            skipped.append(token)
        else:
            groups.append(phoneme_paths)
            rendered_tokens.append(token)
            group_phonemes.append(list(phs))

    if not groups:
        return None, skipped, []

    units = _compute_units(groups, rendered_tokens, group_phonemes)
    wav_bytes = audio.concatenate(groups)
    return wav_bytes, skipped, units


def _compute_units(
    groups: list[list],
    rendered_tokens: list[str],
    group_phonemes: list[list[str] | None],
) -> list[dict]:
    """Walk the groups in output order, summing per-WAV durations to assign
    each rendered token a [start_ms, end_ms] span in the final stitched audio.
    Mirrors `audio.concatenate`'s inter-group silence.

    When a group was rendered from phoneme paths, each phoneme also gets its
    own sub-span under `unit["phonemes"]`. Word-recording groups get an empty
    phonemes list — we have no boundary information for them.
    """
    inter_ms = audio.INTER_WORD_SILENCE_SECONDS * 1000
    cumulative = 0.0
    units: list[dict] = []
    for i, (group, token, phs) in enumerate(zip(groups, rendered_tokens, group_phonemes)):
        unit_start = cumulative
        phoneme_spans: list[dict] = []
        for j, path in enumerate(group):
            samples, _ = audio.read_wav(path)
            d_ms = len(samples) * 1000 / audio.SAMPLE_RATE
            if phs is not None:
                phoneme_spans.append({
                    "symbol": phs[j],
                    "start_ms": int(round(cumulative)),
                    "end_ms": int(round(cumulative + d_ms)),
                })
            cumulative += d_ms
        units.append({
            "token": token,
            "start_ms": int(round(unit_start)),
            "end_ms": int(round(cumulative)),
            "phonemes": phoneme_spans,
        })
        if i < len(groups) - 1:
            cumulative += inter_ms
    return units
