"""Resolver chain for top-level synthesis tokens.

Each resolver returns a unit dict (see `store.UNIT_FIELDS`) for a token, or
None if it cannot handle it. The synthesis loop walks `RESOLVERS` in order and
takes the first non-None result. Add new fallback strategies by appending to
the list.

`Context` carries what the synth knows about the token's surroundings so a
resolver can pick the best of several recorded copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import phonemes
import store


@dataclass(frozen=True)
class Context:
    prev: str | None = None       # previous word in the sentence (normalised)
    next: str | None = None       # next word in the sentence (normalised)
    position: str | None = None   # "mid" | "comma" | "final"


class Resolver(Protocol):
    def resolve(self, token: str, ctx: Context) -> dict | None: ...


class WordResolver:
    """Picks the word copy whose recorded neighbours/position best match."""

    def resolve(self, token: str, ctx: Context) -> dict | None:
        return store.best_unit("word", token, prev=ctx.prev, next=ctx.next,
                               position=ctx.position)


class PhonemeResolver:
    """Resolves a raw ARPABET token (e.g. the user typed `HH`).

    Returns None for anything that is not itself an ARPABET symbol. Phoneme
    slots inside a phonemised word are looked up by synth directly, with
    their neighbouring phonemes as context.
    """

    def resolve(self, token: str, ctx: Context) -> dict | None:
        if not phonemes.is_phoneme(token):
            return None
        return store.best_unit("phoneme", token)


RESOLVERS: list[Resolver] = [WordResolver(), PhonemeResolver()]
