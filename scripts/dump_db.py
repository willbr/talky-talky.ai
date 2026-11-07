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
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        recordings = conn.execute("SELECT id, text, wav_path FROM recordings ORDER BY id").fetchall()
        units = conn.execute("SELECT * FROM units ORDER BY kind DESC, label, id").fetchall()

    def missing(wav_path: str) -> str:
        return "" if (Path(inventory.INVENTORY_DIR) / wav_path).exists() else "  [missing]"

    print(f"schema v{version}")
    print(f"\nrecordings ({len(recordings)})")
    for r in recordings:
        print(f"  {r['id']}  {r['text'][:90]}{missing(r['wav_path'])}")
        for s in store.get_sentences(r["id"]):
            segs = store.get_segments(r["id"], s["idx"])
            print(f"    [{s['idx']}] {s['status']:<8} {s['start_ms']:>6}-{s['end_ms']:<6} ms"
                  f"  words={len(segs['words'])} phonemes={len(segs['phonemes'])}"
                  f"  {s['text'][:60]}")

    for kind in ("word", "phoneme", "diphone"):
        rows = [u for u in units if u["kind"] == kind]
        labels = sorted({u["label"] for u in rows})
        print(f"\n{kind}s ({len(labels)} labels, {len(rows)} copies)")
        for label in labels:
            print(f"  {label}")
            for u in (r for r in rows if r["label"] == label):
                if u["source"] == "sentence":
                    src = f"sentence:{u['source_recording_id']}#{u['source_sentence_idx']}"
                elif u["source_word"]:
                    src = f"{u['source']}:{u['source_word']}"
                else:
                    src = u["source"]
                ctx = f"{u['prev'] or '-'}/{u['next'] or '-'}"
                pos = u["position"] or "-"
                print(f"    {u['id']}  {src:<16} {ctx:<12} {pos:<6} {u['duration_ms']:>5} ms"
                      f"{missing(u['wav_path'])}")


if __name__ == "__main__":
    main()
