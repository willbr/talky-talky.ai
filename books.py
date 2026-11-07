"""Read the downloaded book corpus and sample sentences for the Speak UI."""
from __future__ import annotations

import random
import re
from pathlib import Path

BOOKS_DIR = Path(__file__).parent / "data" / "books"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WS_RE = re.compile(r"\s+")

MIN_SENTENCE_LEN = 30
MAX_SENTENCE_LEN = 250


def list_books() -> list[str]:
    if not BOOKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in BOOKS_DIR.glob("*.txt"))


def _read_sentences(book: str) -> list[str]:
    path = BOOKS_DIR / f"{book}.txt"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    normalised = _WS_RE.sub(" ", text).strip()
    sentences = _SENTENCE_SPLIT_RE.split(normalised)
    return [s for s in sentences if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN]


def sample_sentence(book: str | None = None) -> tuple[str, str] | None:
    """Return (book, sentence) or None if nothing usable is available."""
    books = list_books()
    if not books:
        return None
    if not book or book == "any":
        book = random.choice(books)
    if book not in books:
        return None
    sentences = _read_sentences(book)
    if not sentences:
        return None
    return book, random.choice(sentences)


def sample_passage(
    book: str | None = None, n_min: int = 3, n_max: int = 5,
) -> tuple[str, str] | None:
    """Return (book, passage) of n_min..n_max consecutive sentences, or None."""
    books = list_books()
    if not books:
        return None
    if not book or book == "any":
        book = random.choice(books)
    if book not in books:
        return None
    path = BOOKS_DIR / f"{book}.txt"
    text = _WS_RE.sub(" ", path.read_text(encoding="utf-8")).strip()
    sentences = _SENTENCE_SPLIT_RE.split(text)
    n = random.randint(n_min, n_max)
    ok = [2 <= len(s) <= MAX_SENTENCE_LEN for s in sentences]
    starts = [i for i in range(len(sentences) - n + 1) if all(ok[i:i + n])]
    if not starts:
        return None
    i = random.choice(starts)
    return book, " ".join(sentences[i:i + n])
