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


def path(word: str) -> Path:
    return INVENTORY_DIR / f"{normalise(word)}.wav"


def has(word: str) -> bool:
    return path(word).exists()


def save(word: str, wav_bytes: bytes) -> None:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    path(word).write_bytes(wav_bytes)


def list_words() -> list[str]:
    if not INVENTORY_DIR.is_dir():
        return []
    return sorted(p.stem for p in INVENTORY_DIR.glob("*.wav"))
