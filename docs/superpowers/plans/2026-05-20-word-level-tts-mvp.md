# Word-level concatenative TTS MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page Flask app that records spoken words via the browser microphone and synthesises sentences by stitching the recorded word files together.

**Architecture:** One Flask process serves a single HTML page with a record pane and a synthesise pane. Browser captures raw PCM via AudioWorklet at 22050 Hz and encodes WAV in JavaScript. The server stores each word as a 16-bit mono PCM WAV file in `inventory/<word>.wav` and synthesises by concatenating resolved word files with short fades and inter-word silence. A `WordResolver` is the only resolver in this MVP; the chain leaves a clean seam for a future `PhonemeResolver`.

**Tech Stack:** Python 3.11+, Flask, numpy, stdlib `wave`, pytest. Vanilla JavaScript with the Web Audio API.

**Spec reference:** `docs/superpowers/specs/2026-05-20-word-level-tts-mvp-design.md`

---

## Task 1: Project bootstrap

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Modify: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_sanity.py`

- [ ] **Step 1: Inspect existing .gitignore**

Run: `cat .gitignore` (or `Get-Content .gitignore`)
Expected: shows the current contents. We'll add to it without removing anything.

- [ ] **Step 2: Append to `.gitignore`**

Append these lines to `.gitignore` (preserve existing contents):

```
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
*.egg-info/

# App data
inventory/
*.wav
!tests/fixtures/*.wav
```

The `inventory/` directory will hold user recordings and should never be committed. The `*.wav` rule plus the negation keeps recorded audio out of git while still allowing test fixtures.

- [ ] **Step 3: Create `requirements.txt`**

```
Flask==3.0.3
numpy==2.4.6
```

- [ ] **Step 4: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest==8.3.4
```

- [ ] **Step 5: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 6: Create the virtual environment and install dev dependencies**

Run (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Expected: numpy, Flask, pytest, and their transitive deps install without errors.

- [ ] **Step 7: Create `tests/__init__.py` (empty file)**

Empty file. Makes `tests/` an importable package.

- [ ] **Step 8: Create `tests/test_sanity.py`**

```python
def test_pytest_is_running():
    assert 1 + 1 == 2
```

- [ ] **Step 9: Run the sanity test**

Run: `pytest -v`
Expected: `1 passed`.

- [ ] **Step 10: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt pyproject.toml tests/__init__.py tests/test_sanity.py
git commit -m "Bootstrap project: deps, pytest config, sanity test"
```

---

## Task 2: `audio.py` — constants and WAV round-trip

**Files:**
- Create: `audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write the failing test in `tests/test_audio.py`**

```python
import numpy as np
import audio


def test_write_then_read_preserves_samples(tmp_path):
    samples = np.array([0, 1000, -1000, 32000, -32000, 0], dtype=np.int16)
    path = tmp_path / "round_trip.wav"

    audio.write_wav(path, samples)
    loaded, rate = audio.read_wav(path)

    assert rate == audio.SAMPLE_RATE
    assert loaded.dtype == np.int16
    np.testing.assert_array_equal(loaded, samples)


def test_constants():
    assert audio.SAMPLE_RATE == 22050
    assert audio.CHANNELS == 1
    assert audio.SAMPLE_WIDTH == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_audio.py -v`
Expected: `ModuleNotFoundError: No module named 'audio'`.

- [ ] **Step 3: Implement `audio.py`**

```python
"""Audio I/O and DSP primitives for talky-talky.ai.

Uses stdlib `wave` for I/O and numpy for sample manipulation.
All recordings in the inventory are 16-bit mono PCM at 22050 Hz.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16-bit signed PCM)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    assert samples.dtype == np.int16, f"expected int16, got {samples.dtype}"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as r:
        rate = r.getframerate()
        n = r.getnframes()
        frames = r.readframes(n)
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, rate
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_audio.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add audio.py with WAV round-trip and format constants"
```

---

## Task 3: `audio.py` — `trim_silence`

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Add failing tests to `tests/test_audio.py`**

Append:

```python
def test_trim_silence_removes_padding():
    silence = np.zeros(1000, dtype=np.int16)
    loud = (np.ones(500, dtype=np.int16) * 10000)
    signal = np.concatenate([silence, loud, silence])

    trimmed = audio.trim_silence(signal)

    assert len(trimmed) == 500
    np.testing.assert_array_equal(trimmed, loud)


def test_trim_silence_all_silent_returns_empty():
    signal = np.zeros(5000, dtype=np.int16)
    trimmed = audio.trim_silence(signal)
    assert len(trimmed) == 0


def test_trim_silence_caps_at_max_trim():
    # 2 seconds of silence followed by 100 ms of tone — trim should cap at 500 ms
    silence_samples = audio.SAMPLE_RATE * 2
    tone_samples = int(audio.SAMPLE_RATE * 0.1)
    signal = np.concatenate([
        np.zeros(silence_samples, dtype=np.int16),
        np.ones(tone_samples, dtype=np.int16) * 10000,
    ])

    trimmed = audio.trim_silence(signal)

    max_trim = int(audio.SAMPLE_RATE * 0.5)
    expected_min_len = (silence_samples - max_trim) + tone_samples
    assert len(trimmed) >= expected_min_len
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_audio.py -v`
Expected: three failures with `AttributeError: module 'audio' has no attribute 'trim_silence'`.

- [ ] **Step 3: Add `trim_silence` and threshold constants to `audio.py`**

Add the new constants near the top (after `SAMPLE_WIDTH`):

```python
SILENCE_THRESHOLD_RATIO = 0.02  # fraction of peak below which we consider a sample silent
MAX_TRIM_SECONDS = 0.5
```

Add the function:

```python
def trim_silence(samples: np.ndarray) -> np.ndarray:
    """Trim leading and trailing silence, capped at MAX_TRIM_SECONDS per side.

    A sample counts as "loud" if its absolute value exceeds
    SILENCE_THRESHOLD_RATIO * peak. If every sample is below threshold
    (e.g. all zeros), returns an empty array.
    """
    if len(samples) == 0:
        return samples

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return np.array([], dtype=np.int16)

    threshold = int(peak * SILENCE_THRESHOLD_RATIO)
    loud_mask = np.abs(samples) > threshold
    loud_indices = np.flatnonzero(loud_mask)
    if loud_indices.size == 0:
        return np.array([], dtype=np.int16)

    first_loud = int(loud_indices[0])
    last_loud = int(loud_indices[-1])

    max_trim = int(SAMPLE_RATE * MAX_TRIM_SECONDS)
    start = max(0, first_loud - 0)  # keep first loud sample
    start = max(start, first_loud - max_trim) if first_loud > max_trim else 0
    end = min(len(samples), last_loud + 1)
    end = min(end, last_loud + 1 + max_trim) if (len(samples) - last_loud - 1) > max_trim else len(samples)

    # Apply the cap: never trim more than max_trim from either side.
    cap_start = max(0, first_loud - max_trim)
    cap_end = min(len(samples), last_loud + 1 + max_trim)
    return samples[cap_start:cap_end]
```

Note: the function above keeps the simpler clamping logic and discards the intermediate `start`/`end` variables. Replace the function body with this cleaner version:

```python
def trim_silence(samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return samples

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return np.array([], dtype=np.int16)

    threshold = int(peak * SILENCE_THRESHOLD_RATIO)
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return np.array([], dtype=np.int16)

    first, last = int(loud[0]), int(loud[-1])
    max_trim = int(SAMPLE_RATE * MAX_TRIM_SECONDS)
    start = max(0, first - max_trim)
    end = min(len(samples), last + 1 + max_trim)
    return samples[start:end]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_audio.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add trim_silence with 500ms per-side cap"
```

---

## Task 4: `audio.py` — `peak_normalise`

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_audio.py`:

```python
def test_peak_normalise_scales_to_target():
    samples = np.array([0, 1000, -1000, 500], dtype=np.int16)
    normalised = audio.peak_normalise(samples)
    assert int(np.max(np.abs(normalised))) == audio.PEAK_TARGET


def test_peak_normalise_silent_input_unchanged():
    silent = np.zeros(100, dtype=np.int16)
    out = audio.peak_normalise(silent)
    np.testing.assert_array_equal(out, silent)


def test_peak_target_constant():
    # -3 dBFS relative to int16 full scale
    expected = int(32767 * 10 ** (-3 / 20))
    assert audio.PEAK_TARGET == expected
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_audio.py -v`
Expected: three failures for missing `peak_normalise` and `PEAK_TARGET`.

- [ ] **Step 3: Add `PEAK_TARGET` constant and `peak_normalise` to `audio.py`**

After `MAX_TRIM_SECONDS`, add:

```python
PEAK_TARGET = int(32767 * 10 ** (-3 / 20))  # -3 dBFS in int16 full scale (~23197)
```

After `trim_silence`, add:

```python
def peak_normalise(samples: np.ndarray) -> np.ndarray:
    """Scale samples so the loudest sample sits at PEAK_TARGET. Silent input is returned unchanged."""
    if len(samples) == 0:
        return samples
    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return samples
    scale = PEAK_TARGET / peak
    return (samples.astype(np.float32) * scale).clip(-32768, 32767).astype(np.int16)
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_audio.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add peak_normalise targeting -3 dBFS"
```

---

## Task 5: `audio.py` — `apply_fade`

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_audio.py`:

```python
def test_apply_fade_ramps_start_and_end():
    samples = np.ones(2000, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)

    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    # First sample of fade-in is ~0, ramps up
    assert abs(int(faded[0])) < 100
    assert int(faded[fade_n // 2]) < int(faded[fade_n - 1])
    # Last sample of fade-out is ~0, ramps down
    assert abs(int(faded[-1])) < 100
    assert int(faded[-fade_n // 2]) < int(faded[-fade_n])


def test_apply_fade_leaves_middle_unchanged():
    samples = np.ones(2000, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)

    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    middle = faded[fade_n:-fade_n]
    np.testing.assert_array_equal(middle, samples[fade_n:-fade_n])


def test_apply_fade_short_signal_returns_unchanged():
    # Signal shorter than 2 * fade window — fade can't be applied safely
    fade_n = int(audio.SAMPLE_RATE * audio.FADE_SECONDS)
    samples = np.ones(fade_n, dtype=np.int16) * 10000
    faded = audio.apply_fade(samples)
    np.testing.assert_array_equal(faded, samples)
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_audio.py -v`
Expected: three failures for missing `apply_fade` and `FADE_SECONDS`.

- [ ] **Step 3: Add `FADE_SECONDS` constant and `apply_fade` to `audio.py`**

After `PEAK_TARGET`, add:

```python
FADE_SECONDS = 0.005  # 5 ms linear fade-in/fade-out
```

After `peak_normalise`, add:

```python
def apply_fade(samples: np.ndarray) -> np.ndarray:
    """Apply a 5 ms linear fade-in at the start and fade-out at the end.

    Returns the input unchanged if it is shorter than two fade windows.
    """
    fade_n = int(SAMPLE_RATE * FADE_SECONDS)
    if len(samples) < 2 * fade_n:
        return samples

    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    out = samples.astype(np.float32).copy()
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out.clip(-32768, 32767).astype(np.int16)
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_audio.py -v`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add 5ms linear fades to suppress click artifacts at joins"
```

---

## Task 6: `audio.py` — `concatenate`

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_audio.py`:

```python
import io

def test_concatenate_combines_two_wavs(tmp_path):
    a_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000  # 1 sec
    b_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 5000   # 1 sec
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    audio.write_wav(a_path, a_samples)
    audio.write_wav(b_path, b_samples)

    wav_bytes = audio.concatenate([a_path, b_path])

    # Load result back via stdlib wave to validate header + length
    loaded, rate = audio.read_wav(io.BytesIO(wav_bytes))
    assert rate == audio.SAMPLE_RATE
    expected_silence = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    expected_len = len(a_samples) + expected_silence + len(b_samples)
    assert len(loaded) == expected_len


def test_concatenate_single_path_no_silence_appended(tmp_path):
    samples = np.ones(1000, dtype=np.int16) * 10000
    path = tmp_path / "only.wav"
    audio.write_wav(path, samples)

    wav_bytes = audio.concatenate([path])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == len(samples)


def test_concatenate_empty_list_returns_empty_wav():
    wav_bytes = audio.concatenate([])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == 0
```

The test uses `io.BytesIO(wav_bytes)` with `read_wav`. Update `read_wav` to accept either a path or a file-like object (next step covers this).

- [ ] **Step 2: Update `read_wav` to accept a path or a binary stream**

In `audio.py`, replace the `read_wav` body:

```python
def read_wav(source) -> tuple[np.ndarray, int]:
    """Read a WAV file. `source` may be a path or a binary file-like object."""
    if hasattr(source, "read"):
        ctx = wave.open(source, "rb")
    else:
        ctx = wave.open(str(source), "rb")
    with ctx as r:
        rate = r.getframerate()
        n = r.getnframes()
        frames = r.readframes(n)
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, rate
```

- [ ] **Step 3: Run to verify the new tests fail**

Run: `pytest tests/test_audio.py -v`
Expected: `concatenate`-related failures only; the existing tests still pass.

- [ ] **Step 4: Add `INTER_WORD_SILENCE_SECONDS` constant and `concatenate` to `audio.py`**

After `FADE_SECONDS`, add:

```python
INTER_WORD_SILENCE_SECONDS = 0.030
```

At the bottom of `audio.py`, add:

```python
def concatenate(paths: list[Path]) -> bytes:
    """Read each WAV at `paths`, apply fade, append with inter-word silence, return WAV bytes."""
    import io as _io

    silence_n = int(SAMPLE_RATE * INTER_WORD_SILENCE_SECONDS)
    silence = np.zeros(silence_n, dtype=np.int16)

    chunks: list[np.ndarray] = []
    for i, p in enumerate(paths):
        samples, _ = read_wav(p)
        samples = apply_fade(samples)
        chunks.append(samples)
        if i < len(paths) - 1:
            chunks.append(silence)

    output = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(output.tobytes())
    return buf.getvalue()
```

- [ ] **Step 5: Run to verify passes**

Run: `pytest tests/test_audio.py -v`
Expected: `14 passed`.

- [ ] **Step 6: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add concatenate with 30ms inter-word silence and per-clip fades"
```

---

## Task 7: `inventory.py` — `normalise`

**Files:**
- Create: `inventory.py`
- Create: `tests/test_inventory.py`

- [ ] **Step 1: Write failing tests in `tests/test_inventory.py`**

```python
import inventory


def test_normalise_lowercases():
    assert inventory.normalise("Hello") == "hello"


def test_normalise_strips_punctuation():
    assert inventory.normalise("Hello!") == "hello"
    assert inventory.normalise("Don't") == "dont"
    assert inventory.normalise("(world)") == "world"


def test_normalise_strips_surrounding_whitespace():
    assert inventory.normalise("  hello  ") == "hello"


def test_normalise_empty_input_returns_empty():
    assert inventory.normalise("") == ""
    assert inventory.normalise("...") == ""
    assert inventory.normalise("!?") == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_inventory.py -v`
Expected: `ModuleNotFoundError: No module named 'inventory'`.

- [ ] **Step 3: Implement `inventory.py`**

```python
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
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_inventory.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add inventory.py tests/test_inventory.py
git commit -m "Add inventory.normalise (lowercase + strip ASCII punctuation)"
```

---

## Task 8: `inventory.py` — `save`, `has`, `list_words`, `path`

**Files:**
- Modify: `inventory.py`
- Modify: `tests/test_inventory.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inventory.py`:

```python
import pytest


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def test_path_uses_normalised_filename(temp_inventory):
    assert inventory.path("Hello!") == temp_inventory / "hello.wav"


def test_save_writes_bytes_to_disk(temp_inventory):
    wav_bytes = b"RIFF....fakewav....data...."
    inventory.save("hello", wav_bytes)
    assert (temp_inventory / "hello.wav").read_bytes() == wav_bytes


def test_has_reflects_existence(temp_inventory):
    assert not inventory.has("hello")
    inventory.save("hello", b"x")
    assert inventory.has("hello")


def test_list_words_returns_sorted_unique(temp_inventory):
    inventory.save("world", b"x")
    inventory.save("hello", b"x")
    inventory.save("apple", b"x")
    assert inventory.list_words() == ["apple", "hello", "world"]


def test_save_creates_inventory_dir_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "newdir"
    monkeypatch.setattr(inventory, "INVENTORY_DIR", target)
    inventory.save("hello", b"x")
    assert target.is_dir()
    assert (target / "hello.wav").exists()
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_inventory.py -v`
Expected: multiple failures for missing `path`, `save`, `has`, `list_words`.

- [ ] **Step 3: Add functions to `inventory.py`**

At the end of `inventory.py`:

```python
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
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_inventory.py -v`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add inventory.py tests/test_inventory.py
git commit -m "Add inventory save/has/list_words/path with on-disk persistence"
```

---

## Task 9: `resolver.py` — `WordResolver`

**Files:**
- Create: `resolver.py`
- Create: `tests/test_resolver.py`

- [ ] **Step 1: Write failing tests in `tests/test_resolver.py`**

```python
import pytest

import inventory
import resolver


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def test_word_resolver_hit(temp_inventory):
    inventory.save("hello", b"fakewav")
    r = resolver.WordResolver()
    assert r.resolve("hello") == temp_inventory / "hello.wav"


def test_word_resolver_miss(temp_inventory):
    r = resolver.WordResolver()
    assert r.resolve("missing") is None


def test_word_resolver_normalises_input(temp_inventory):
    inventory.save("hello", b"fakewav")
    r = resolver.WordResolver()
    assert r.resolve("Hello!") == temp_inventory / "hello.wav"


def test_resolver_chain_contains_word_resolver():
    assert any(isinstance(r, resolver.WordResolver) for r in resolver.RESOLVERS)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_resolver.py -v`
Expected: `ModuleNotFoundError: No module named 'resolver'`.

- [ ] **Step 3: Implement `resolver.py`**

```python
"""Resolver chain for synthesis lookups.

Each resolver returns a `Path` to a WAV file for a given word, or `None` if
it cannot handle the word. The synthesis loop walks `RESOLVERS` in order and
takes the first non-None result. This is the seam for future phoneme fallback:
add a `PhonemeResolver` to the list and synthesis loop never changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import inventory


class Resolver(Protocol):
    def resolve(self, word: str) -> Path | None: ...


class WordResolver:
    """Resolves a word by looking up `inventory/<normalised>.wav`."""

    def resolve(self, word: str) -> Path | None:
        return inventory.path(word) if inventory.has(word) else None


RESOLVERS: list[Resolver] = [WordResolver()]
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_resolver.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add resolver.py tests/test_resolver.py
git commit -m "Add resolver chain with WordResolver (seam for future PhonemeResolver)"
```

---

## Task 10: `synth.py` — `tokenise`

**Files:**
- Create: `synth.py`
- Create: `tests/test_synth.py`

- [ ] **Step 1: Write failing tests in `tests/test_synth.py`**

```python
import synth


def test_tokenise_splits_and_lowercases():
    assert synth.tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    assert synth.tokenise("Hello, world!") == ["hello", "world"]
    assert synth.tokenise("Don't stop.") == ["dont", "stop"]


def test_tokenise_collapses_whitespace():
    assert synth.tokenise("  hello   world  ") == ["hello", "world"]


def test_tokenise_empty_returns_empty_list():
    assert synth.tokenise("") == []
    assert synth.tokenise("   ") == []
    assert synth.tokenise("...!?") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_synth.py -v`
Expected: `ModuleNotFoundError: No module named 'synth'`.

- [ ] **Step 3: Implement `tokenise` in `synth.py`**

```python
"""Sentence synthesis: tokenise -> resolve -> concatenate."""
from __future__ import annotations

import audio
import inventory
import resolver


def tokenise(sentence: str) -> list[str]:
    """Split sentence into normalised tokens. Empty tokens are dropped."""
    raw_tokens = sentence.split()
    tokens = [inventory.normalise(t) for t in raw_tokens]
    return [t for t in tokens if t]
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_synth.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add synth.py tests/test_synth.py
git commit -m "Add synth.tokenise (matches inventory normalisation rules)"
```

---

## Task 11: `synth.py` — `synthesise`

**Files:**
- Modify: `synth.py`
- Modify: `tests/test_synth.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_synth.py`:

```python
import io

import numpy as np
import pytest

import audio
import inventory


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _make_recording(path, n_samples=2000, amplitude=10000):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    audio.write_wav(path, samples)


def test_synthesise_all_words_present(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")
    _make_recording(temp_inventory / "world.wav")

    wav_bytes, skipped = synth.synthesise("hello world")

    assert wav_bytes is not None
    assert skipped == []
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) > 0


def test_synthesise_some_missing(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")

    wav_bytes, skipped = synth.synthesise("hello mars")

    assert wav_bytes is not None
    assert skipped == ["mars"]


def test_synthesise_all_missing(temp_inventory):
    wav_bytes, skipped = synth.synthesise("mars venus")

    assert wav_bytes is None
    assert skipped == ["mars", "venus"]


def test_synthesise_case_and_punctuation_insensitive(temp_inventory):
    _make_recording(temp_inventory / "hello.wav")
    _make_recording(temp_inventory / "world.wav")

    wav_bytes, skipped = synth.synthesise("Hello, WORLD!")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_empty_sentence(temp_inventory):
    wav_bytes, skipped = synth.synthesise("")
    assert wav_bytes is None
    assert skipped == []
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_synth.py -v`
Expected: failures for `AttributeError: module 'synth' has no attribute 'synthesise'`.

- [ ] **Step 3: Implement `synthesise` in `synth.py`**

Append to `synth.py`:

```python
def synthesise(sentence: str) -> tuple[bytes | None, list[str]]:
    """Synthesise a sentence by stitching recorded word files.

    Returns (wav_bytes_or_None, skipped_tokens). `wav_bytes` is None when
    nothing in the sentence could be resolved (either the sentence is empty
    or every token is missing).
    """
    tokens = tokenise(sentence)
    if not tokens:
        return None, []

    resolved_paths = []
    skipped: list[str] = []
    for token in tokens:
        path = None
        for r in resolver.RESOLVERS:
            path = r.resolve(token)
            if path is not None:
                break
        if path is None:
            skipped.append(token)
        else:
            resolved_paths.append(path)

    if not resolved_paths:
        return None, skipped

    wav_bytes = audio.concatenate(resolved_paths)
    return wav_bytes, skipped
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_synth.py -v`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add synth.py tests/test_synth.py
git commit -m "Add synth.synthesise walking resolver chain per token"
```

---

## Task 12: `app.py` — `GET /` and `GET /api/inventory`

**Files:**
- Create: `app.py`
- Create: `templates/index.html` (minimal placeholder)
- Create: `tests/test_app.py`

- [ ] **Step 1: Create a placeholder `templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>talky-talky.ai</title></head>
<body><div id="app">talky-talky.ai</div></body>
</html>
```

(We'll flesh this out in Task 15.)

- [ ] **Step 2: Write failing tests in `tests/test_app.py`**

```python
import pytest

import app as app_module
import inventory


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"talky-talky.ai" in r.data


def test_inventory_empty(client):
    r = client.get("/api/inventory")
    assert r.status_code == 200
    assert r.get_json() == {"words": []}


def test_inventory_lists_saved_words(client, tmp_path):
    (tmp_path / "hello.wav").write_bytes(b"x")
    (tmp_path / "apple.wav").write_bytes(b"x")
    r = client.get("/api/inventory")
    assert r.status_code == 200
    assert r.get_json() == {"words": ["apple", "hello"]}
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_app.py -v`
Expected: `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 4: Implement `app.py` with `create_app`, `GET /`, `GET /api/inventory`**

```python
"""Flask app factory and routes for talky-talky.ai."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template

import inventory

log = logging.getLogger("talky")


def create_app() -> Flask:
    flask_app = Flask(__name__, static_folder="static", template_folder="templates")
    logging.basicConfig(level=logging.INFO)

    @flask_app.route("/")
    def index() -> str:
        return render_template("index.html")

    @flask_app.route("/api/inventory")
    def api_inventory():
        return jsonify({"words": inventory.list_words()})

    return flask_app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
```

- [ ] **Step 5: Run to verify passes**

Run: `pytest tests/test_app.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add app.py templates/index.html tests/test_app.py
git commit -m "Add Flask app factory with GET / and GET /api/inventory"
```

---

## Task 13: `app.py` — `POST /api/record`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_app.py`:

```python
import io
import wave

import numpy as np

import audio


def _make_wav_bytes(n_samples=audio.SAMPLE_RATE, amplitude=10000, sample_rate=audio.SAMPLE_RATE):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def test_record_saves_valid_wav(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record",
        data={"word": "Hello!", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.get_json() == {"word": "hello"}
    assert (tmp_path / "hello.wav").exists()


def test_record_rejects_wrong_sample_rate(client):
    wav = _make_wav_bytes(sample_rate=44100)
    r = client.post(
        "/api/record",
        data={"word": "hello", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_rejects_empty_normalised_word(client):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record",
        data={"word": "...", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_rejects_too_short(client):
    wav = _make_wav_bytes(n_samples=100)  # ~5 ms — well below 100 ms threshold
    r = client.post(
        "/api/record",
        data={"word": "hi", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_409_on_existing_word(client, tmp_path):
    (tmp_path / "hello.wav").write_bytes(b"existing")
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record",
        data={"word": "hello", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 409


def test_record_overwrite_true_replaces(client, tmp_path):
    (tmp_path / "hello.wav").write_bytes(b"existing")
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record?overwrite=true",
        data={"word": "hello", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert (tmp_path / "hello.wav").read_bytes() != b"existing"
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_app.py -v`
Expected: 405 or 404 for missing route on all six new tests.

- [ ] **Step 3: Add `MIN_LENGTH_SECONDS` to `audio.py`**

After `INTER_WORD_SILENCE_SECONDS`:

```python
MIN_LENGTH_SECONDS = 0.1
```

- [ ] **Step 4: Add `POST /api/record` to `app.py`**

Add imports at the top:

```python
import io
import wave

import numpy as np

import audio
from flask import request
```

(Adjust the existing import block to include `request` from flask.)

Add the route inside `create_app`, after `api_inventory`:

```python
    @flask_app.route("/api/record", methods=["POST"])
    def api_record():
        word_raw = request.form.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            log.warning("record: empty word after normalisation: %r", word_raw)
            return jsonify({"error": "word is empty after normalisation"}), 400

        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "missing audio field"}), 400

        raw = audio_file.read()
        try:
            with wave.open(io.BytesIO(raw), "rb") as r:
                rate = r.getframerate()
                channels = r.getnchannels()
                sampwidth = r.getsampwidth()
                n_frames = r.getnframes()
                frames = r.readframes(n_frames)
        except wave.Error as e:
            log.warning("record: invalid WAV: %s", e)
            return jsonify({"error": "invalid WAV file"}), 400

        if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
            log.warning(
                "record: format mismatch rate=%s channels=%s sampwidth=%s",
                rate, channels, sampwidth,
            )
            return jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400

        samples = np.frombuffer(frames, dtype=np.int16)
        samples = audio.trim_silence(samples)
        if len(samples) < int(audio.SAMPLE_RATE * audio.MIN_LENGTH_SECONDS):
            log.warning("record: too short after trim: %s samples", len(samples))
            return jsonify({"error": "recording too short"}), 400

        samples = audio.peak_normalise(samples)

        overwrite = request.args.get("overwrite") == "true"
        if inventory.has(word) and not overwrite:
            return jsonify({"error": "word exists", "existing": word}), 409

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(audio.CHANNELS)
            w.setsampwidth(audio.SAMPLE_WIDTH)
            w.setframerate(audio.SAMPLE_RATE)
            w.writeframes(samples.tobytes())
        inventory.save(word, out.getvalue())

        log.info("record: saved %r (%d samples)", word, len(samples))
        return jsonify({"word": word}), 200
```

- [ ] **Step 5: Run to verify passes**

Run: `pytest tests/test_app.py -v`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add app.py audio.py tests/test_app.py
git commit -m "Add POST /api/record with format validation and silence trim"
```

---

## Task 14: `app.py` — `POST /api/synthesize`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_app.py`:

```python
import base64


def test_synthesize_returns_audio_and_skipped(client, tmp_path):
    # Seed inventory with hello and world
    for word in ("hello", "world"):
        samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
        audio.write_wav(tmp_path / f"{word}.wav", samples)

    r = client.post("/api/synthesize", json={"sentence": "Hello, world!"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["skipped"] == []
    assert data["audio"] is not None
    decoded = base64.b64decode(data["audio"])
    assert decoded.startswith(b"RIFF")


def test_synthesize_reports_skipped(client, tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "hello.wav", samples)

    r = client.post("/api/synthesize", json={"sentence": "hello mars"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["skipped"] == ["mars"]
    assert data["audio"] is not None


def test_synthesize_all_missing(client):
    r = client.post("/api/synthesize", json={"sentence": "mars venus"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["audio"] is None
    assert data["skipped"] == ["mars", "venus"]


def test_synthesize_empty_sentence_400(client):
    r = client.post("/api/synthesize", json={"sentence": ""})
    assert r.status_code == 400


def test_synthesize_missing_field_400(client):
    r = client.post("/api/synthesize", json={})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_app.py -v`
Expected: failures on the five new tests.

- [ ] **Step 3: Add `POST /api/synthesize` to `app.py`**

Add the import at the top:

```python
import base64

import synth
```

Add the route inside `create_app`, after `api_record`:

```python
    @flask_app.route("/api/synthesize", methods=["POST"])
    def api_synthesize():
        data = request.get_json(silent=True) or {}
        sentence = data.get("sentence")
        if sentence is None:
            return jsonify({"error": "missing 'sentence' field"}), 400
        if not sentence.strip():
            return jsonify({"error": "sentence is empty"}), 400

        wav_bytes, skipped = synth.synthesise(sentence)
        encoded = base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None
        log.info("synthesize: %r resolved=%s skipped=%s",
                 sentence, "yes" if wav_bytes else "no", skipped)
        return jsonify({"audio": encoded, "skipped": skipped}), 200
```

- [ ] **Step 4: Run to verify passes**

Run: `pytest tests/test_app.py -v`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Add POST /api/synthesize returning base64 WAV and skipped tokens"
```

---

## Task 15: `templates/index.html` and `static/style.css` — two-pane layout

**Files:**
- Modify: `templates/index.html`
- Create: `static/style.css`

This task has no automated tests — it is HTML and CSS only. Manual check confirms the layout.

- [ ] **Step 1: Replace `templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>talky-talky.ai</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <header>
    <h1>talky-talky.ai</h1>
    <p class="tag">basic concatenative TTS</p>
  </header>

  <section id="inventory-bar">
    <h2>Inventory</h2>
    <div id="inventory-chips" aria-live="polite"></div>
  </section>

  <main>
    <section id="record-pane">
      <h2>Record a word</h2>
      <label>Word
        <input id="word-input" type="text" autocomplete="off" placeholder="hello">
      </label>
      <div class="controls">
        <button id="record-btn">● Record</button>
        <button id="stop-btn" disabled>■ Stop</button>
        <button id="save-btn" disabled>Save</button>
        <button id="discard-btn" disabled>Discard</button>
      </div>
      <audio id="preview" controls></audio>
      <p id="record-status" role="status"></p>
    </section>

    <section id="synth-pane">
      <h2>Synthesise a sentence</h2>
      <textarea id="sentence-input" rows="3" placeholder="hello world"></textarea>
      <button id="synth-btn">Synthesise</button>
      <audio id="output" controls></audio>
      <div id="skipped-list" aria-live="polite"></div>
    </section>
  </main>

  <script src="{{ url_for('static', filename='app.js') }}" type="module"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/style.css`**

```css
:root {
  --bg: #1a1b1e;
  --panel: #25272c;
  --fg: #e6e6e6;
  --muted: #9aa0a6;
  --accent: #4ea8de;
  --error: #ef6262;
  --radius: 8px;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 1.5rem;
  background: var(--bg);
  color: var(--fg);
  font: 15px/1.4 system-ui, sans-serif;
}

header h1 { margin: 0; }
header .tag { color: var(--muted); margin-top: 0.2rem; }

#inventory-bar {
  margin: 1.5rem 0;
  padding: 1rem;
  background: var(--panel);
  border-radius: var(--radius);
}

#inventory-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.5rem;
}

.chip {
  background: #34373d;
  color: var(--fg);
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.85rem;
}

.chip.missing {
  background: #4a2e2e;
  color: var(--error);
  cursor: pointer;
}

main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

section { background: var(--panel); padding: 1.2rem; border-radius: var(--radius); }
section h2 { margin-top: 0; }

label { display: block; margin-bottom: 0.8rem; }
input, textarea {
  width: 100%;
  background: #1a1b1e;
  color: var(--fg);
  border: 1px solid #3a3d44;
  border-radius: 4px;
  padding: 0.5rem;
  font: inherit;
}

button {
  background: var(--accent);
  color: #001019;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  font: inherit;
  cursor: pointer;
  margin-right: 0.4rem;
}
button:disabled { background: #3a3d44; color: var(--muted); cursor: not-allowed; }

.controls { margin-bottom: 0.8rem; }

audio { display: block; width: 100%; margin: 0.6rem 0; }

#record-status { color: var(--muted); margin: 0.4rem 0 0 0; min-height: 1.3em; }

#skipped-list { margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
```

- [ ] **Step 3: Smoke check the layout in a browser**

Run (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
python -m flask --app app run
```

Open `http://127.0.0.1:5000` in a browser. Expected: two panes side by side, empty inventory bar, all buttons disabled except Record. Stop the server with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add templates/index.html static/style.css
git commit -m "Add two-pane layout and dark stylesheet"
```

---

## Task 16: `static/recorder-worklet.js` and `static/app.js` — recording

**Files:**
- Create: `static/recorder-worklet.js`
- Create: `static/app.js`

This task has no automated tests; verification is manual in the browser.

- [ ] **Step 1: Create `static/recorder-worklet.js`**

```javascript
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffers = [];
    this.recording = false;
    this.port.onmessage = (e) => {
      if (e.data === 'start') {
        this.buffers = [];
        this.recording = true;
      } else if (e.data === 'stop') {
        this.recording = false;
        let total = 0;
        for (const b of this.buffers) total += b.length;
        const out = new Float32Array(total);
        let off = 0;
        for (const b of this.buffers) { out.set(b, off); off += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this.buffers = [];
      }
    };
  }

  process(inputs) {
    if (this.recording && inputs[0] && inputs[0][0]) {
      this.buffers.push(new Float32Array(inputs[0][0]));
    }
    return true;
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
```

- [ ] **Step 2: Create `static/app.js`**

```javascript
const SAMPLE_RATE = 22050;

const els = {
  wordInput: document.getElementById('word-input'),
  recordBtn: document.getElementById('record-btn'),
  stopBtn: document.getElementById('stop-btn'),
  saveBtn: document.getElementById('save-btn'),
  discardBtn: document.getElementById('discard-btn'),
  preview: document.getElementById('preview'),
  recordStatus: document.getElementById('record-status'),
  inventoryChips: document.getElementById('inventory-chips'),
  sentenceInput: document.getElementById('sentence-input'),
  synthBtn: document.getElementById('synth-btn'),
  output: document.getElementById('output'),
  skippedList: document.getElementById('skipped-list'),
};

let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let currentWavBlob = null;

async function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule('/static/recorder-worklet.js');
}

async function startRecording() {
  await ensureAudio();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
  source.connect(workletNode);
  workletNode.port.postMessage('start');
  setStatus('recording…');
  setButtons({ record: false, stop: true, save: false, discard: false });
}

function stopRecording() {
  return new Promise((resolve) => {
    workletNode.port.onmessage = (e) => {
      const float32 = e.data;
      mediaStream.getTracks().forEach((t) => t.stop());
      workletNode.disconnect();
      mediaStream = null;
      workletNode = null;
      resolve(float32);
    };
    workletNode.port.postMessage('stop');
  });
}

function floatsToInt16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    let s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function encodeWav(int16, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = int16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < int16.length; i++, offset += 2) view.setInt16(offset, int16[i], true);

  return new Blob([buffer], { type: 'audio/wav' });
}

function setStatus(text, isError = false) {
  els.recordStatus.textContent = text;
  els.recordStatus.style.color = isError ? 'var(--error)' : 'var(--muted)';
}

function setButtons({ record, stop, save, discard }) {
  els.recordBtn.disabled = !record;
  els.stopBtn.disabled = !stop;
  els.saveBtn.disabled = !save;
  els.discardBtn.disabled = !discard;
}

async function refreshInventory() {
  const r = await fetch('/api/inventory');
  const { words } = await r.json();
  els.inventoryChips.innerHTML = '';
  for (const w of words) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = w;
    els.inventoryChips.appendChild(chip);
  }
}

els.recordBtn.addEventListener('click', async () => {
  try {
    await startRecording();
  } catch (err) {
    setStatus('mic permission denied or unavailable', true);
    setButtons({ record: true, stop: false, save: false, discard: false });
  }
});

els.stopBtn.addEventListener('click', async () => {
  const floats = await stopRecording();
  const int16 = floatsToInt16(floats);
  currentWavBlob = encodeWav(int16, SAMPLE_RATE);
  els.preview.src = URL.createObjectURL(currentWavBlob);
  setStatus(`captured ${(floats.length / SAMPLE_RATE).toFixed(2)}s — review then Save`);
  setButtons({ record: true, stop: false, save: true, discard: true });
});

els.discardBtn.addEventListener('click', () => {
  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setStatus('discarded');
  setButtons({ record: true, stop: false, save: false, discard: false });
});

els.saveBtn.addEventListener('click', async () => {
  const word = els.wordInput.value.trim();
  if (!word) { setStatus('enter a word first', true); return; }
  if (!currentWavBlob) { setStatus('record something first', true); return; }

  const form = new FormData();
  form.append('word', word);
  form.append('audio', currentWavBlob, 'audio.wav');

  let r = await fetch('/api/record', { method: 'POST', body: form });
  if (r.status === 409) {
    if (!confirm(`"${word}" already exists. Overwrite?`)) return;
    r = await fetch('/api/record?overwrite=true', { method: 'POST', body: form });
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return;
  }
  const { word: saved } = await r.json();
  setStatus(`saved "${saved}"`);
  currentWavBlob = null;
  els.preview.removeAttribute('src');
  els.wordInput.value = '';
  setButtons({ record: true, stop: false, save: false, discard: false });
  await refreshInventory();
});

refreshInventory();
setButtons({ record: true, stop: false, save: false, discard: false });
```

- [ ] **Step 3: Manual verification — record a word**

Run (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
python -m flask --app app run
```

Open `http://127.0.0.1:5000`. Allow microphone access when prompted.
1. Type `hello` into the word input.
2. Click Record, speak the word, click Stop.
3. Verify the preview `<audio>` plays back what you said.
4. Click Save. Verify the inventory chip list shows "hello".
5. Confirm `inventory/hello.wav` exists on disk and is a valid WAV (e.g., open in any audio app).

Stop the server with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add static/recorder-worklet.js static/app.js
git commit -m "Add browser recording: AudioWorklet capture, JS WAV encoder, save flow"
```

---

## Task 17: `static/app.js` — synthesis UI and skipped-chip wiring

**Files:**
- Modify: `static/app.js`

- [ ] **Step 1: Append synthesis logic to `static/app.js`**

Append at the bottom of `static/app.js`:

```javascript
async function synthesise() {
  const sentence = els.sentenceInput.value.trim();
  if (!sentence) return;

  const r = await fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentence }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    els.skippedList.textContent = `synth failed: ${err.error || r.statusText}`;
    return;
  }
  const { audio: b64, skipped } = await r.json();

  if (b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/wav' });
    els.output.src = URL.createObjectURL(blob);
    els.output.play().catch(() => { /* user gesture is the click on synth */ });
  } else {
    els.output.removeAttribute('src');
  }

  renderSkipped(skipped);
}

function renderSkipped(skipped) {
  els.skippedList.innerHTML = '';
  if (!skipped.length) return;

  const label = document.createElement('span');
  label.textContent = 'Skipped (click to record):';
  label.style.color = 'var(--muted)';
  label.style.marginRight = '0.4rem';
  els.skippedList.appendChild(label);

  for (const w of skipped) {
    const chip = document.createElement('span');
    chip.className = 'chip missing';
    chip.textContent = w;
    chip.addEventListener('click', () => {
      els.wordInput.value = w;
      els.wordInput.focus();
    });
    els.skippedList.appendChild(chip);
  }
}

els.synthBtn.addEventListener('click', synthesise);
```

- [ ] **Step 2: Manual verification — full record-then-synthesise loop**

Run (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
python -m flask --app app run
```

Open `http://127.0.0.1:5000`.
1. Record "hello" (if not already saved).
2. Record "world".
3. In the synthesise pane, type `hello world` and click Synthesise.
4. Verify the output `<audio>` plays the two words concatenated.
5. Type `hello mars` and synthesise. Verify "mars" appears as a clickable red chip under Skipped, and clicking the chip prefills the word input.
6. Type `mars venus`. Verify the skipped list shows both and no audio plays.

Stop the server with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "Add synthesis UI with playback and clickable skipped-word chips"
```

---

## Task 18: README updates and acceptance run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the "Getting started" section of `README.md`**

Find the section beginning `## Getting started` and replace it with:

```markdown
## Getting started

Requires Python 3.11+ and a modern desktop browser (Chrome or Firefox) that supports the Web Audio API and AudioWorklet.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m flask --app app run
```

Open `http://127.0.0.1:5000`. Record a few words, then type a sentence to synthesise it.

Run tests with `pytest`.
```

- [ ] **Step 2: Acceptance run**

Run (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
pytest -v
```
Expected: all tests pass (around 30+ tests total across modules).

```powershell
python -m flask --app app run
```

Open the app and run through the full workflow:
1. Record "hello", "world", "the", "quick".
2. Synthesise `the quick hello world`. Verify audio playback contains all four words.
3. Synthesise `the lazy fox`. Verify "lazy" and "fox" appear as skipped chips and the audio plays "the" only.
4. Click the "lazy" chip — verify it prefills the word input.
5. Stop the server.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Update README with getting-started instructions for the MVP"
```

---

## Self-review

**Spec coverage check:**

- ✅ Single Flask app, single page, two panes — Tasks 12 + 15.
- ✅ Filesystem inventory at `inventory/<word>.wav` — Task 8.
- ✅ Audio constants (22050 Hz, mono, 16-bit) — Task 2.
- ✅ Trim silence with 500 ms cap — Task 3.
- ✅ Peak normalisation to -3 dBFS — Task 4.
- ✅ 5 ms fade-in/out at joins — Task 5.
- ✅ 30 ms inter-word silence — Task 6.
- ✅ Tokenisation (lowercase, strip punctuation) — Tasks 7, 10.
- ✅ `WordResolver` + extensible chain — Task 9.
- ✅ `synthesise` returns audio + skipped — Task 11.
- ✅ `GET /api/inventory` — Task 12.
- ✅ `POST /api/record` with format validation, 409 on conflict, 400 on too-short — Task 13.
- ✅ `POST /api/synthesize` returning base64 WAV + skipped — Task 14.
- ✅ Browser AudioWorklet capture + JS WAV encoder — Task 16.
- ✅ Skipped-words clickable chips that pre-fill recorder — Task 17.
- ✅ Tests for each backend module — every backend task includes tests.
- ✅ Out-of-scope items (browser tests, audio quality assertions, mobile UI) — explicitly excluded.

**Placeholder scan:** No TBDs, no "add appropriate error handling", every code block is complete.

**Type consistency:** `synthesise` signature `tuple[bytes | None, list[str]]` is used consistently between Task 11 and Task 14. `WordResolver.resolve(word) -> Path | None` matches `synth.synthesise` usage. `inventory.normalise`/`save`/`has`/`list_words`/`path` signatures match between Tasks 7, 8, 9, 12, 13.

**Plan complete.**

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-word-level-tts-mvp.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
