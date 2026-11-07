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


PHONEME_SUBDIR = "phonemes"
LEGACY_VARIANT_ID = "00000000T000000000000"
# Variant IDs are `%Y%m%dT%H%M%S%f` timestamps. Anything else (notably
# path separators or `..`) is rejected so IDs can't escape the symbol dir.
_VARIANT_ID_RE = re.compile(r"\d{8}T\d{12}")


def is_valid_id(variant_id: str) -> bool:
    return bool(_VARIANT_ID_RE.fullmatch(variant_id))


def _phoneme_dir() -> Path:
    return INVENTORY_DIR / PHONEME_SUBDIR


def _symbol_dir(symbol: str) -> Path:
    return _phoneme_dir() / symbol.lower()


def _variant_id() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _migrate_legacy_phoneme(symbol: str) -> None:
    """Move a pre-existing flat `inventory/phonemes/<symbol>.wav` into the
    per-symbol subdir as `<LEGACY_VARIANT_ID>.wav` with a stub sidecar.

    Idempotent: skips if the flat file is absent or already migrated.
    """
    import json
    import wave

    flat = _phoneme_dir() / f"{symbol.lower()}.wav"
    if not flat.exists():
        return
    sym_dir = _symbol_dir(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    target_wav = sym_dir / f"{LEGACY_VARIANT_ID}.wav"
    target_json = sym_dir / f"{LEGACY_VARIANT_ID}.json"
    if target_wav.exists():
        # Should not happen, but be defensive — just drop the flat file.
        try:
            flat.unlink()
        except OSError:
            pass
        return
    flat.rename(target_wav)
    try:
        with wave.open(str(target_wav), "rb") as r:
            duration_ms = int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, OSError):
        duration_ms = 0
    target_json.write_text(json.dumps({
        "id": LEGACY_VARIANT_ID,
        "symbol": symbol.upper(),
        "source": "isolated",
        "prev": None,
        "next": None,
        "duration_ms": duration_ms,
    }), encoding="utf-8")


def save_variant(
    symbol: str,
    wav_bytes: bytes,
    *,
    source: str,
    prev: str | None,
    next: str | None,
) -> str:
    """Write a new variant for the phoneme. Returns the new variant ID."""
    import io
    import json
    import wave

    sym_dir = _symbol_dir(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    vid = _variant_id()
    wav_path = sym_dir / f"{vid}.wav"
    wav_path.write_bytes(wav_bytes)
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            duration_ms = int(r.getnframes() * 1000 / r.getframerate())
    except (wave.Error, EOFError):
        duration_ms = 0
    (sym_dir / f"{vid}.json").write_text(json.dumps({
        "id": vid,
        "symbol": symbol.upper(),
        "source": source,
        "prev": prev,
        "next": next,
        "duration_ms": duration_ms,
    }), encoding="utf-8")
    return vid


def list_variants(symbol: str) -> list[dict]:
    """Return all variant metadata for `symbol`, oldest first.

    Runs the lazy legacy migration on first call when a flat
    `inventory/phonemes/<symbol>.wav` is still present.
    """
    import json

    _migrate_legacy_phoneme(symbol)
    sym_dir = _symbol_dir(symbol)
    if not sym_dir.is_dir():
        return []
    variants: list[dict] = []
    for j in sorted(sym_dir.glob("*.json")):
        try:
            variants.append(json.loads(j.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return variants


def variant_path(symbol: str, variant_id: str) -> Path:
    return _symbol_dir(symbol) / f"{variant_id}.wav"


def delete_variant(symbol: str, variant_id: str) -> bool:
    if not is_valid_id(variant_id):
        return False
    wav = variant_path(symbol, variant_id)
    meta = wav.with_suffix(".json")
    if not wav.exists() and not meta.exists():
        return False
    try:
        wav.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        meta.unlink(missing_ok=True)
    except OSError:
        pass
    return True


def best_phoneme_variant(
    symbol: str,
    prev: str | None = None,
    next: str | None = None,
) -> Path | None:
    """Pick the variant whose recorded prev/next neighbours best match the
    target context. Score = (prev_match ? 1 : 0) + (next_match ? 1 : 0);
    highest wins, tiebreak by most-recent ID. Returns None when no
    variant exists for the symbol."""
    variants = list_variants(symbol)
    if not variants:
        return None

    def score(v: dict) -> tuple[int, str]:
        s = 0
        if v.get("prev") == prev:
            s += 1
        if v.get("next") == next:
            s += 1
        return (s, v.get("id", ""))

    best = max(variants, key=score)
    return variant_path(symbol, best["id"])


def has_phoneme(symbol: str) -> bool:
    return bool(list_variants(symbol))


def list_recorded_phonemes() -> list[str]:
    """Return upper-cased symbols that have at least one variant on disk.

    Triggers legacy migration for any flat `<symbol>.wav` files via
    list_variants — so this also acts as a sweep on first access.
    """
    d = _phoneme_dir()
    if not d.is_dir():
        return []
    # Eagerly migrate any leftover flat files so they show up under their
    # symbol's subdir below.
    for flat in list(d.glob("*.wav")):
        _migrate_legacy_phoneme(flat.stem)
    recorded: list[str] = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and any(sub.glob("*.wav")):
            recorded.append(sub.name.upper())
    return recorded


SENTENCE_SUBDIR = "sentences"


def _sentence_dir() -> Path:
    return INVENTORY_DIR / SENTENCE_SUBDIR


def save_sentence(text: str, wav_bytes: bytes) -> str:
    """Persist a (text, audio) pair for future DL training. Returns the id."""
    from datetime import datetime, timezone
    sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    d = _sentence_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.wav").write_bytes(wav_bytes)
    (d / f"{sid}.txt").write_text(text, encoding="utf-8")
    return sid


def find_sentence_by_text(text: str) -> Path | None:
    """Return the path to the most recent reference WAV matching `text`
    by normalised content (lowercase, punctuation stripped, whitespace
    collapsed — same rule the synth tokeniser uses). None if no match.
    """
    needle = normalise(text)
    if not needle:
        return None
    d = _sentence_dir()
    if not d.is_dir():
        return None
    matches: list[Path] = []
    for txt in d.glob("*.txt"):
        try:
            if normalise(txt.read_text(encoding="utf-8")) == needle:
                wav = txt.with_suffix(".wav")
                if wav.exists():
                    matches.append(wav)
        except OSError:
            continue
    if not matches:
        return None
    # Latest takes priority — IDs are UTC timestamps so lexicographic sort
    # is chronological.
    matches.sort(key=lambda p: p.stem)
    return matches[-1]


def list_sentences() -> list[dict]:
    d = _sentence_dir()
    if not d.is_dir():
        return []
    items = []
    for wav in sorted(d.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        items.append({"id": wav.stem, "text": txt.read_text(encoding="utf-8")})
    return items
