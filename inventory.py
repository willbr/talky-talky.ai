"""Inventory location and word normalisation.

All recordings live under `INVENTORY_DIR`; their metadata is in the SQLite
store (see `store.py`). The normalisation rule is shared with synthesis
tokenisation: lowercase ASCII letters and digits, punctuation stripped,
whitespace collapsed.
"""
from __future__ import annotations

import re
from pathlib import Path

INVENTORY_DIR = Path("inventory")

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

# Unit and sentence IDs are `%Y%m%dT%H%M%S%f` timestamps. Anything else
# (notably path separators or `..`) is rejected so IDs can't escape audio/.
_ID_RE = re.compile(r"\d{8}T\d{12}")


def normalise(word: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Returns "" for empty results."""
    lower = word.lower().strip()
    stripped = _PUNCT_RE.sub("", lower)
    return re.sub(r"\s+", " ", stripped).strip()


def is_valid_id(unit_id: object) -> bool:
    return isinstance(unit_id, str) and bool(_ID_RE.fullmatch(unit_id))
