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
