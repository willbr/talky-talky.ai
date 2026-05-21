"""Download a small corpus of public-domain classics from Project Gutenberg.

Run via `uv run python scripts/download_books.py`. Files land in
`data/books/` and are gitignored.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("download_books")

BOOKS: dict[str, str] = {
    "treasure-island.txt": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
    "alice-in-wonderland.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "pride-and-prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "tom-sawyer.txt": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    "frankenstein.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "books"


def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return text
    body_start = text.find("\n", start_idx) + 1
    return text[body_start:end_idx].strip() + "\n"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "talky-talky.ai/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in BOOKS.items():
        target = OUTPUT_DIR / filename
        if target.exists():
            log.info("skip %s (already downloaded)", filename)
            continue
        log.info("fetching %s", url)
        try:
            raw = fetch(url)
        except Exception as e:
            log.warning("failed %s: %s", filename, e)
            continue
        body = strip_gutenberg_boilerplate(raw)
        target.write_text(body, encoding="utf-8")
        log.info("saved %s (%d bytes)", target, len(body))


if __name__ == "__main__":
    main()
