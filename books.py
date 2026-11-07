"""Read the downloaded book corpus and sample sentences for the Speak UI."""
from __future__ import annotations

import random
import re
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _all_sentences(book: str) -> tuple[str, ...]:
    """Every sentence of a book, cached per process (books don't change)."""
    path = BOOKS_DIR / f"{book}.txt"
    if not path.is_file():
        return ()
    text = _WS_RE.sub(" ", path.read_text(encoding="utf-8")).strip()
    return tuple(_SENTENCE_SPLIT_RE.split(text))


@lru_cache(maxsize=None)
def _passage_starts(book: str, n: int) -> tuple[int, ...]:
    sentences = _all_sentences(book)
    ok = [2 <= len(s) <= MAX_SENTENCE_LEN for s in sentences]
    return tuple(i for i in range(len(sentences) - n + 1) if all(ok[i:i + n]))


def _read_sentences(book: str) -> list[str]:
    return [s for s in _all_sentences(book) if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN]


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
    sentences = _all_sentences(book)
    n = random.randint(n_min, n_max)
    starts = _passage_starts(book, n)
    if not starts:
        return None
    i = random.choice(starts)
    return book, " ".join(sentences[i:i + n])
