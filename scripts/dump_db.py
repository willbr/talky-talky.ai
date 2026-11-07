"""Print the inventory database as readable text.

Usage: uv run python scripts/dump_db.py
"""
from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inventory  # noqa: E402
import store  # noqa: E402


def main() -> None:
    with closing(store.connect()) as conn:
        sentences = conn.execute("SELECT id, text, wav_path FROM sentences ORDER BY id").fetchall()
        units = conn.execute("SELECT * FROM units ORDER BY kind DESC, label, id").fetchall()

    def missing(wav_path: str) -> str:
        return "" if (Path(inventory.INVENTORY_DIR) / wav_path).exists() else "  [missing]"

    print(f"sentences ({len(sentences)})")
    for s in sentences:
        print(f"  {s['id']}  {s['text']}{missing(s['wav_path'])}")

    for kind in ("word", "phoneme"):
        rows = [u for u in units if u["kind"] == kind]
        labels = sorted({u["label"] for u in rows})
        print(f"\n{kind}s ({len(labels)} labels, {len(rows)} copies)")
        for label in labels:
            print(f"  {label}")
            for u in (r for r in rows if r["label"] == label):
                src = u["source"] if not u["source_word"] else f"{u['source']}:{u['source_word']}"
                ctx = f"{u['prev'] or '-'}/{u['next'] or '-'}"
                pos = u["position"] or "-"
                print(f"    {u['id']}  {src:<16} {ctx:<12} {pos:<6} {u['duration_ms']:>5} ms"
                      f"{missing(u['wav_path'])}")


if __name__ == "__main__":
    main()
