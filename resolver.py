"""Resolver chain for synthesis lookups.

Each resolver returns a `Path` to a WAV file for a given word, or `None` if
it cannot handle the word. The synthesis loop walks `RESOLVERS` in order and
takes the first non-None result. This is the seam for future phoneme fallback:
add a `PhonemeResolver` to the list and synthesis loop never changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import inventory
import phonemes


class Resolver(Protocol):
    def resolve(self, word: str) -> Path | None: ...


class WordResolver:
    """Resolves a word by looking up `inventory/<normalised>.wav`."""

    def resolve(self, word: str) -> Path | None:
        return inventory.path(word) if inventory.has(word) else None


class PhonemeResolver:
    """Resolves an ARPABET phoneme token to its recorded WAV file.

    Returns None for any token that is not itself a valid ARPABET symbol,
    even if a file by that name exists in the word inventory. The synthesis
    loop is responsible for phonemising English words; this resolver only
    handles raw phoneme symbols and so calls best_phoneme_variant without
    any context (prev=next=None).
    """

    def resolve(self, word: str) -> Path | None:
        if not phonemes.is_phoneme(word):
            return None
        return inventory.best_phoneme_variant(word)


RESOLVERS: list[Resolver] = [WordResolver(), PhonemeResolver()]
