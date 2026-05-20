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


class Resolver(Protocol):
    def resolve(self, word: str) -> Path | None: ...


class WordResolver:
    """Resolves a word by looking up `inventory/<normalised>.wav`."""

    def resolve(self, word: str) -> Path | None:
        return inventory.path(word) if inventory.has(word) else None


RESOLVERS: list[Resolver] = [WordResolver()]
