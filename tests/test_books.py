import pytest

import books


@pytest.fixture
def books_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(books, "BOOKS_DIR", tmp_path)
    return tmp_path


def test_list_books_empty(books_dir):
    assert books.list_books() == []


def test_list_books_returns_stems(books_dir):
    (books_dir / "alice.txt").write_text("x", encoding="utf-8")
    (books_dir / "frankenstein.txt").write_text("y", encoding="utf-8")
    assert books.list_books() == ["alice", "frankenstein"]


def test_sample_sentence_none_when_empty(books_dir):
    assert books.sample_sentence() is None


def test_sample_sentence_returns_from_specific_book(books_dir):
    text = "Short. " + ("A real sentence with enough characters to pass the minimum length filter. " * 3)
    (books_dir / "alice.txt").write_text(text, encoding="utf-8")
    result = books.sample_sentence("alice")
    assert result is not None
    book, sentence = result
    assert book == "alice"
    assert 30 <= len(sentence) <= 250


def test_sample_sentence_unknown_book_returns_none(books_dir):
    (books_dir / "alice.txt").write_text(
        "A real sentence with enough characters to pass the minimum length filter. ",
        encoding="utf-8",
    )
    assert books.sample_sentence("does-not-exist") is None


def test_sample_sentence_filters_too_short_and_too_long(books_dir):
    # Only short sentences — should return None.
    (books_dir / "tiny.txt").write_text("Hi. Yo. Eh.", encoding="utf-8")
    assert books.sample_sentence("tiny") is None
