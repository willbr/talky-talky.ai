"""Inventory of recorded words on disk.

Each word is stored as `<INVENTORY_DIR>/<normalised>.wav`. The normalisation
rule is shared with synthesis tokenisation: lowercase ASCII letters and digits,
all punctuation stripped, surrounding whitespace removed.
"""
from __future__ import annotations

import re
from pathlib import Path

INVENTORY_DIR = Path("inventory")

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalise(word: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Returns "" for empty results."""
    lower = word.lower().strip()
    stripped = _PUNCT_RE.sub("", lower)
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return collapsed
