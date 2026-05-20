"""Sentence synthesis: tokenise -> resolve -> concatenate."""
from __future__ import annotations

import audio
import inventory
import resolver


def tokenise(sentence: str) -> list[str]:
    """Split sentence into normalised tokens. Empty tokens are dropped."""
    raw_tokens = sentence.split()
    tokens = [inventory.normalise(t) for t in raw_tokens]
    return [t for t in tokens if t]


def synthesise(sentence: str) -> tuple[bytes | None, list[str]]:
    """Synthesise a sentence by stitching recorded word files.

    Returns (wav_bytes_or_None, skipped_tokens). `wav_bytes` is None when
    nothing in the sentence could be resolved (either the sentence is empty
    or every token is missing).
    """
    tokens = tokenise(sentence)
    if not tokens:
        return None, []

    resolved_paths = []
    skipped: list[str] = []
    for token in tokens:
        path = None
        for r in resolver.RESOLVERS:
            path = r.resolve(token)
            if path is not None:
                break
        if path is None:
            skipped.append(token)
        else:
            resolved_paths.append(path)

    if not resolved_paths:
        return None, skipped

    wav_bytes = audio.concatenate(resolved_paths)
    return wav_bytes, skipped
