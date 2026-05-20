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


def synthesise(sentence: str) -> tuple[bytes | None, list[str]]:
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

    Returns (wav_bytes_or_None, skipped_tokens).
    """
    tokens = tokenise(sentence)
    if not tokens:
        return None, []

    groups: list[list] = []
    skipped: list[str] = []

    for token in tokens:
        # Pass 1: try the chain on the raw token.
        direct = None
        for r in resolver.RESOLVERS:
            direct = r.resolve(token)
            if direct is not None:
                break

        if direct is not None:
            groups.append([direct])
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
            for r in resolver.RESOLVERS:
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

    if not groups:
        return None, skipped

    wav_bytes = audio.concatenate(groups)
    return wav_bytes, skipped
