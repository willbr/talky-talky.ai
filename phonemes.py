"""ARPABET phoneme set and grapheme-to-phoneme via CMUdict.

`PHONEMES` is the canonical 39-symbol set used in CMUdict. Stress digit
suffixes (e.g. `AH1`) are NOT phonemes — they are dropped at dict-load time.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

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

_DICT_PATH = Path(__file__).parent / "data" / "cmudict.dict"
_STRESS_RE = re.compile(r"\d+$")


def _strip_stress(phoneme: str) -> str:
    return _STRESS_RE.sub("", phoneme).upper()


def _load_dict() -> dict[str, list[str]]:
    """Parse `data/cmudict.dict` into a `{lowercase_word: [PHONEMES…]}` dict.

    Rules:
      - Lines starting with `;;;` are comments — skip.
      - Tokens after the word that don't strip down to a known ARPABET symbol
        cause the whole entry to be skipped (with a WARNING log).
      - Entries marked as alternate pronunciations (`word(2)`, `word(3)`, ...)
        are skipped — first wins.
    """
    if not _DICT_PATH.is_file():
        raise FileNotFoundError(
            f"CMUdict not found at {_DICT_PATH}. "
            "Run Task 1 of the phoneme-fallback plan to bundle it."
        )

    d: dict[str, list[str]] = {}
    with _DICT_PATH.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";;;"):
                continue
            # Strip inline comments (rare, format-dependent).
            if " #" in line:
                line = line.split(" #", 1)[0].rstrip()

            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0]
            if "(" in word:
                # Alternate pronunciation — skip; first one wins.
                continue
            word_lc = word.lower()
            if word_lc in d:
                continue  # already have an entry (defensive)

            phs: list[str] | None = []
            for p in parts[1:]:
                stripped = _strip_stress(p)
                if stripped not in PHONEMES_SET:
                    log.warning(
                        "cmudict: unknown phoneme %r in entry %r — skipping entry",
                        p, word_lc,
                    )
                    phs = None
                    break
                phs.append(stripped)

            if phs is not None:
                d[word_lc] = phs
    return d


_DICT: dict[str, list[str]] = _load_dict()


def word_to_phonemes(word: str) -> list[str] | None:
    """Look up `word` in CMUdict and return its ARPABET phoneme list (case-insensitive).

    Returns `None` if the word is not in the dictionary, or if the input is
    empty or whitespace-only.
    """
    if word is None:
        return None
    key = word.strip().lower()
    if not key:
        return None
    phs = _DICT.get(key)
    return list(phs) if phs is not None else None


def is_phoneme(token: str) -> bool:
    """Return True iff `token` is an ARPABET phoneme symbol (case-insensitive)."""
    if not token:
        return False
    return token.upper() in PHONEMES_SET


def list_phonemes() -> list[str]:
    """Return the canonical phoneme list (in declaration order)."""
    return list(PHONEMES)
