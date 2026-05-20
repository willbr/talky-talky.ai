"""ARPABET phoneme set and grapheme-to-phoneme via CMUdict.

`PHONEMES` is the canonical 39-symbol set used in CMUdict. Stress digit
suffixes (e.g. `AH1`) are NOT phonemes — they are dropped at dict-load time.
"""
from __future__ import annotations

PHONEMES: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH",
    "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
)

PHONEMES_SET: frozenset[str] = frozenset(PHONEMES)


def is_phoneme(token: str) -> bool:
    """Return True iff `token` is an ARPABET phoneme symbol (case-insensitive)."""
    if not token:
        return False
    return token.upper() in PHONEMES_SET


def list_phonemes() -> list[str]:
    """Return the canonical phoneme list (in declaration order)."""
    return list(PHONEMES)
