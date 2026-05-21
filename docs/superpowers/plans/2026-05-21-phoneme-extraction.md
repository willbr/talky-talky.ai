# Phoneme Extraction (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the user records a word, automatically propose duration-weighted phoneme boundaries from the CMUdict sequence and offer a waveform editor where they can refine boundaries and choose which slices to save into the phoneme inventory.

**Architecture:** New `extraction.py` module owns the duration-prior allocation math. A new `audio.slice_samples` helper does the cut. Two new Flask routes (`POST /api/extract/propose`, `POST /api/extract/save`) wrap them; the propose route returns the word's base64 audio + initial boundaries, save accepts edited boundaries and writes the selected phoneme files. The frontend builds a waveform editor dynamically in JS — canvas rendering, draggable boundary handles, per-phoneme play and save toggles.

**Tech Stack:** Python 3.13 via uv, Flask, numpy, stdlib `wave`. Vanilla JavaScript with the Web Audio API and a 2D canvas.

**Spec reference:** `docs/superpowers/specs/2026-05-21-phoneme-extraction-design.md`

**Project conventions reminder:**
- Run Python via `uv run …` only.
- Commit directly to `master`; no feature branches.
- Tests: `uv run pytest -v`.

---

## Task 1: `extraction.py` — duration table and `propose_boundaries`

**Files:**
- Create: `extraction.py`
- Create: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing tests in `tests/test_extraction.py`**

```python
import logging

import numpy as np

import audio
import extraction
import phonemes


def _tone(duration_ms: int, amplitude: int = 10000) -> np.ndarray:
    n = int(audio.SAMPLE_RATE * duration_ms / 1000)
    return np.ones(n, dtype=np.int16) * amplitude


def _silence(duration_ms: int) -> np.ndarray:
    n = int(audio.SAMPLE_RATE * duration_ms / 1000)
    return np.zeros(n, dtype=np.int16)


def test_duration_table_covers_every_phoneme():
    assert set(extraction.PHONEME_DURATION_MS.keys()) == phonemes.PHONEMES_SET
    assert all(isinstance(v, int) and v > 0 for v in extraction.PHONEME_DURATION_MS.values())


def test_propose_boundaries_basic_returns_n_plus_one_monotonic():
    samples = _tone(800)
    bounds = extraction.propose_boundaries(samples, ["HH", "AH", "L", "OW"])
    assert len(bounds) == 5
    for i in range(len(bounds) - 1):
        assert bounds[i] <= bounds[i + 1]


def test_propose_boundaries_proportional_to_priors():
    # 460 ms of loud audio, AA (150) + K (80) -> total prior 230, span 460 -> first split at 300 ms.
    samples = _tone(460)
    bounds = extraction.propose_boundaries(samples, ["AA", "K"])
    assert bounds[0] == 0
    assert bounds[2] == 460
    # vowel:stop should be ~150:80 within the slice widths, tolerate ±5 ms rounding.
    vowel_width = bounds[1] - bounds[0]
    stop_width = bounds[2] - bounds[1]
    assert abs(vowel_width - 300) <= 5
    assert abs(stop_width - 160) <= 5


def test_propose_boundaries_skips_leading_silence():
    samples = np.concatenate([_silence(200), _tone(600)])
    bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert 190 <= bounds[0] <= 210     # ~200 ms in
    assert bounds[-1] == 800           # full audio = 800 ms


def test_propose_boundaries_skips_trailing_silence():
    samples = np.concatenate([_tone(600), _silence(300)])
    bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert bounds[0] == 0
    assert 590 <= bounds[-1] <= 610    # last loud sample ~600 ms in


def test_propose_boundaries_all_silent_fallback(caplog):
    samples = _silence(500)
    with caplog.at_level(logging.WARNING, logger="extraction"):
        bounds = extraction.propose_boundaries(samples, ["HH", "AH"])
    assert bounds[0] == 0
    assert bounds[-1] == 500
    assert any("silent" in r.message.lower() for r in caplog.records)


def test_propose_boundaries_unknown_phoneme_uses_default(caplog):
    samples = _tone(400)
    with caplog.at_level(logging.WARNING, logger="extraction"):
        bounds = extraction.propose_boundaries(samples, ["FAKE", "AH"])
    # Allocation still produces N+1 boundaries; default duration is the documented 100 ms.
    assert len(bounds) == 3
    assert any("unknown phoneme" in r.message.lower() for r in caplog.records)


def test_propose_boundaries_empty_sequence_returns_single_zero():
    samples = _tone(400)
    assert extraction.propose_boundaries(samples, []) == [0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: `ModuleNotFoundError: No module named 'extraction'`.

- [ ] **Step 3: Implement `extraction.py`**

```python
"""Duration-weighted phoneme boundary proposal.

Pure math; no I/O. Operates on int16 samples and a phoneme sequence,
returns N+1 boundary times in milliseconds for N phonemes.

The first and last boundaries are clamped to the first/last LOUD sample
(peak * 2%), so silent edges are excluded by default. Word-internal silences
are not detected; users fix them by dragging in the editor.
"""
from __future__ import annotations

import logging

import numpy as np

import phonemes

log = logging.getLogger("extraction")

SAMPLE_RATE = 22050
SILENCE_THRESHOLD_RATIO = 0.02   # same threshold the audio.trim_silence uses
DEFAULT_DURATION_MS = 100        # fallback for unknown phoneme symbols

PHONEME_DURATION_MS: dict[str, int] = {
    # vowels (~150 ms)
    "AA": 150, "AE": 150, "AH": 150, "AO": 150, "AW": 150, "AY": 150,
    "EH": 150, "ER": 150, "EY": 150, "IH": 150, "IY": 150, "OW": 150,
    "OY": 150, "UH": 150, "UW": 150,
    # stops (~80 ms)
    "B": 80, "D": 80, "G": 80, "K": 80, "P": 80, "T": 80,
    # nasals (~90 ms)
    "M": 90, "N": 90, "NG": 90,
    # fricatives (~120 ms)
    "DH": 120, "F": 120, "HH": 120, "S": 120, "SH": 120,
    "TH": 120, "V": 120, "Z": 120, "ZH": 120,
    # liquids (~90 ms)
    "L": 90, "R": 90,
    # glides (~80 ms)
    "W": 80, "Y": 80,
    # affricates (~130 ms)
    "CH": 130, "JH": 130,
}


def _loud_region_ms(samples: np.ndarray, sample_rate: int) -> tuple[int, int]:
    """Return (start_ms, end_ms) covering the loud region of `samples`.

    Fallback: if everything is below threshold, returns the full audio span
    and logs a WARNING.
    """
    duration_ms = int(len(samples) * 1000 / sample_rate)
    if len(samples) == 0:
        return 0, 0

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        log.warning("propose_boundaries: all samples silent; using full audio range")
        return 0, duration_ms

    threshold = int(peak * SILENCE_THRESHOLD_RATIO)
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        log.warning("propose_boundaries: no samples above threshold; using full audio range")
        return 0, duration_ms

    first, last = int(loud[0]), int(loud[-1])
    start_ms = int(first * 1000 / sample_rate)
    end_ms = int((last + 1) * 1000 / sample_rate)
    return start_ms, end_ms


def propose_boundaries(
    samples: np.ndarray,
    phoneme_seq: list[str],
    sample_rate: int = SAMPLE_RATE,
) -> list[int]:
    """Return N+1 boundary times in milliseconds for N phonemes."""
    n = len(phoneme_seq)
    if n == 0:
        return [0]

    start_ms, end_ms = _loud_region_ms(samples, sample_rate)
    span = max(0, end_ms - start_ms)

    priors: list[int] = []
    for p in phoneme_seq:
        sym = p.upper()
        if sym in PHONEME_DURATION_MS:
            priors.append(PHONEME_DURATION_MS[sym])
        else:
            log.warning("propose_boundaries: unknown phoneme %r; using default %d ms",
                        p, DEFAULT_DURATION_MS)
            priors.append(DEFAULT_DURATION_MS)

    total_prior = sum(priors)
    if total_prior == 0:
        # Pathological — shouldn't happen since defaults are positive.
        return [start_ms] + [start_ms] * (n - 1) + [end_ms]

    bounds = [start_ms]
    cumulative = 0
    for prior in priors[:-1]:
        cumulative += prior
        bounds.append(start_ms + int(cumulative * span / total_prior))
    bounds.append(end_ms)
    return bounds
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Confirm full suite is unaffected**

Run: `uv run pytest -v`
Expected: 95 prior + 8 new = `103 passed`.

- [ ] **Step 6: Commit**

```bash
git add extraction.py tests/test_extraction.py
git commit -m "Add extraction.py with duration-weighted phoneme boundary proposal"
```

---

## Task 2: `audio.py` — `slice_samples` helper

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Append failing tests to `tests/test_audio.py`**

```python
def test_slice_samples_basic():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)  # exactly 1 second
    out = audio.slice_samples(s, 100, 200)
    expected_len = int(audio.SAMPLE_RATE * 0.1)  # 100 ms worth
    assert len(out) == expected_len


def test_slice_samples_clamps_out_of_range():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)  # 1 second
    out = audio.slice_samples(s, -50, 1500)
    assert len(out) == len(s)


def test_slice_samples_inverted_range_returns_empty():
    s = np.ones(audio.SAMPLE_RATE, dtype=np.int16)
    out = audio.slice_samples(s, 500, 200)
    assert len(out) == 0
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 18 prior passed + 3 new failures (`AttributeError: module 'audio' has no attribute 'slice_samples'`).

- [ ] **Step 3: Add `slice_samples` to `audio.py`**

Append at the end of `audio.py`:

```python
def slice_samples(
    samples: np.ndarray,
    start_ms: int,
    end_ms: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return the samples between two millisecond timestamps.

    Out-of-range inputs are clamped to the valid sample range. An inverted
    range (end_ms <= start_ms) returns an empty array.
    """
    start_i = max(0, int(start_ms * sample_rate / 1000))
    end_i = min(len(samples), int(end_ms * sample_rate / 1000))
    if end_i <= start_i:
        return np.array([], dtype=samples.dtype)
    return samples[start_i:end_i]
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_audio.py -v`
Expected: `21 passed`.

- [ ] **Step 5: Confirm full suite still green**

Run: `uv run pytest -v`
Expected: 103 + 3 = `106 passed`.

- [ ] **Step 6: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Add audio.slice_samples helper"
```

---

## Task 3: `app.py` — `POST /api/extract/propose`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Append failing tests to `tests/test_app.py`**

```python
import extraction  # noqa: F401  (just to ensure the module is importable in test env)


def test_extract_propose_returns_audio_and_boundaries(client, tmp_path):
    # Seed a word recording so propose has something to work with.
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "hello.wav", samples)

    r = client.post("/api/extract/propose", json={"word": "hello"})
    assert r.status_code == 200
    data = r.get_json()

    assert data["word"] == "hello"
    assert data["audio_duration_ms"] > 0
    assert data["audio"].startswith("UklG") or data["audio"]  # base64-RIFF starts with "UklG"; fall back to non-empty
    # Every phoneme entry has symbol/example/recorded
    assert len(data["phonemes"]) >= 1
    for entry in data["phonemes"]:
        assert "symbol" in entry
        assert "example" in entry
        assert "recorded" in entry
        assert isinstance(entry["recorded"], bool)
    # Boundaries = N + 1 phoneme slots, monotonic non-decreasing
    assert len(data["boundaries_ms"]) == len(data["phonemes"]) + 1
    for i in range(len(data["boundaries_ms"]) - 1):
        assert data["boundaries_ms"][i] <= data["boundaries_ms"][i + 1]


def test_extract_propose_404_when_word_not_recorded(client):
    r = client.post("/api/extract/propose", json={"word": "ghost"})
    assert r.status_code == 404


def test_extract_propose_400_when_word_oov(client, tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "xyzqqqzz.wav", samples)
    r = client.post("/api/extract/propose", json={"word": "xyzqqqzz"})
    assert r.status_code == 400
    body = r.get_json()
    assert "dictionary" in body["error"].lower()


def test_extract_propose_400_when_word_missing_field(client):
    r = client.post("/api/extract/propose", json={})
    assert r.status_code == 400


def test_extract_propose_normalises_word(client, tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "hello.wav", samples)
    r = client.post("/api/extract/propose", json={"word": "  HELLO!  "})
    assert r.status_code == 200
    assert r.get_json()["word"] == "hello"
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_app.py -v`
Expected: prior tests passing + 5 new failures (404/405 — route doesn't exist; the OOV test will likely 404 too).

- [ ] **Step 3: Add `import extraction` to `app.py`**

Add `import extraction` to the first-party imports block of `app.py`, alongside `import audio`, `import inventory`, `import phonemes`, `import synth`. Keep them in alphabetical order.

- [ ] **Step 4: Add the propose route to `app.py`**

Inside `create_app()`, after the existing `api_phoneme_inventory` route handler and BEFORE `api_synthesize`, add:

```python
    @flask_app.route("/api/extract/propose", methods=["POST"])
    def api_extract_propose():
        data = request.get_json(silent=True) or {}
        word_raw = data.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            return jsonify({"error": "word is required"}), 400
        if not inventory.has(word):
            return jsonify({"error": "word not recorded", "word": word}), 404

        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary", "word": word}), 400

        samples, _ = audio.read_wav(inventory.path(word))
        audio_duration_ms = int(len(samples) * 1000 / audio.SAMPLE_RATE)
        boundaries_ms = extraction.propose_boundaries(samples, phs)
        wav_bytes = inventory.path(word).read_bytes()
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        phoneme_payload = [
            {
                "symbol": p,
                "example": phonemes.PHONEME_EXAMPLES.get(p, ""),
                "recorded": inventory.has_phoneme(p),
            }
            for p in phs
        ]

        log.info("extract/propose: word=%r phonemes=%d duration=%dms",
                 word, len(phs), audio_duration_ms)
        return jsonify({
            "word": word,
            "audio_duration_ms": audio_duration_ms,
            "audio": audio_b64,
            "phonemes": phoneme_payload,
            "boundaries_ms": boundaries_ms,
        }), 200
```

- [ ] **Step 5: Run to verify passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: `29 passed` (24 prior + 5 new).

- [ ] **Step 6: Confirm full suite**

Run: `uv run pytest -v`
Expected: 106 + 5 = `111 passed`.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Add POST /api/extract/propose"
```

---

## Task 4: `app.py` — `POST /api/extract/save`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Append failing tests to `tests/test_app.py`**

```python
def _seed_word(tmp_path, name, duration_seconds=1.0, amplitude=10000):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    samples = np.ones(n, dtype=np.int16) * amplitude
    audio.write_wav(tmp_path / f"{name}.wav", samples)


def test_extract_save_writes_selected_phonemes(client, tmp_path):
    _seed_word(tmp_path, "hello")
    # Propose first to get the right phoneme count + a plausible boundary set.
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n_phonemes = len(prop["phonemes"])
    save_flags = [True] * n_phonemes

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": prop["boundaries_ms"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    # All phonemes saved (assuming each slice is >= 30 ms — they will be at 1s/N)
    assert set(body["saved"]) == set(entry["symbol"] for entry in prop["phonemes"])
    assert body["skipped"] == []
    # Files exist on disk
    for entry in prop["phonemes"]:
        assert (tmp_path / "phonemes" / f"{entry['symbol'].lower()}.wav").exists()


def test_extract_save_skips_phonemes_with_save_false(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n_phonemes = len(prop["phonemes"])
    save_flags = [True] * n_phonemes
    save_flags[1] = False  # skip the second phoneme

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": prop["boundaries_ms"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    skipped_symbol = prop["phonemes"][1]["symbol"]
    assert skipped_symbol not in body["saved"]
    # That phoneme file should NOT exist (unless it was already there).
    assert not (tmp_path / "phonemes" / f"{skipped_symbol.lower()}.wav").exists()


def test_extract_save_skips_short_slices(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n = len(prop["phonemes"])
    # Force the second slice to be tiny (1 ms wide) — under the 30 ms floor.
    bounds = list(prop["boundaries_ms"])
    if n >= 2:
        # Make bounds[2] = bounds[1] + 1 (1 ms slice for phoneme index 1).
        bounds[2] = bounds[1] + 1
        # Keep monotonic non-decreasing for the rest.
        for i in range(2, len(bounds) - 1):
            if bounds[i + 1] < bounds[i]:
                bounds[i + 1] = bounds[i]
    save_flags = [True] * n

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": bounds,
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    skipped_symbols = [s["symbol"] for s in body["skipped"]]
    skipped_reasons = [s["reason"] for s in body["skipped"]]
    assert prop["phonemes"][1]["symbol"] in skipped_symbols
    assert "too short" in skipped_reasons


def test_extract_save_rejects_non_monotonic_boundaries(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    bounds = list(prop["boundaries_ms"])
    if len(bounds) >= 3:
        bounds[1], bounds[2] = bounds[2], bounds[1]  # swap to make non-monotonic

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": bounds,
        "save": [True] * (len(bounds) - 1),
    })
    assert r.status_code == 400


def test_extract_save_rejects_length_mismatch(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": prop["boundaries_ms"],
        "save": [True],  # length 1, expected len(boundaries) - 1
    })
    assert r.status_code == 400


def test_extract_save_overwrites_existing_phoneme(client, tmp_path):
    _seed_word(tmp_path, "hello")
    # Pre-populate the first phoneme file with junk.
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    first_symbol = prop["phonemes"][0]["symbol"]
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    target = tmp_path / "phonemes" / f"{first_symbol.lower()}.wav"
    target.write_bytes(b"OLDOLDOLDOLD")

    save_flags = [False] * len(prop["phonemes"])
    save_flags[0] = True  # overwrite just the first one

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": prop["boundaries_ms"],
        "save": save_flags,
    })
    assert r.status_code == 200
    assert target.read_bytes() != b"OLDOLDOLDOLD"


def test_extract_save_404_when_word_not_recorded(client):
    r = client.post("/api/extract/save", json={
        "word": "ghost",
        "boundaries_ms": [0, 100, 200],
        "save": [True, True],
    })
    assert r.status_code == 404


def test_extract_save_normalises_word(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    r = client.post("/api/extract/save", json={
        "word": "  HELLO!  ",
        "boundaries_ms": prop["boundaries_ms"],
        "save": [True] * len(prop["phonemes"]),
    })
    assert r.status_code == 200
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_app.py -v`
Expected: 29 prior passed + 8 new failures (404/405).

- [ ] **Step 3: Add the save route to `app.py`**

Inside `create_app()`, immediately AFTER `api_extract_propose` and BEFORE `api_synthesize`, add:

```python
    @flask_app.route("/api/extract/save", methods=["POST"])
    def api_extract_save():
        data = request.get_json(silent=True) or {}
        word_raw = data.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            return jsonify({"error": "word is required"}), 400
        if not inventory.has(word):
            return jsonify({"error": "word not recorded", "word": word}), 404

        boundaries_ms = data.get("boundaries_ms")
        save_flags = data.get("save")
        if not isinstance(boundaries_ms, list):
            return jsonify({"error": "boundaries_ms is required"}), 400
        if not isinstance(save_flags, list):
            return jsonify({"error": "save is required"}), 400
        if len(boundaries_ms) < 2:
            return jsonify({"error": "boundaries_ms must have at least two entries"}), 400
        if len(save_flags) != len(boundaries_ms) - 1:
            return jsonify({"error": "save length must equal boundaries_ms length minus one"}), 400

        # Monotonic non-decreasing.
        for i in range(len(boundaries_ms) - 1):
            if boundaries_ms[i + 1] < boundaries_ms[i]:
                return jsonify({"error": "boundaries must be non-decreasing"}), 400

        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary"}), 400
        if len(phs) != len(save_flags):
            return jsonify({"error": "phoneme count mismatch"}), 400

        samples, _ = audio.read_wav(inventory.path(word))
        audio_duration_ms = int(len(samples) * 1000 / audio.SAMPLE_RATE)
        for b in boundaries_ms:
            if not isinstance(b, (int, float)) or b < 0 or b > audio_duration_ms + 5:
                return jsonify({"error": "boundary out of range"}), 400

        min_slice_samples = int(audio.SAMPLE_RATE * 0.030)
        saved: list[str] = []
        skipped: list[dict] = []
        for i, should_save in enumerate(save_flags):
            if not should_save:
                continue
            symbol = phs[i]
            slice_arr = audio.slice_samples(samples, int(boundaries_ms[i]), int(boundaries_ms[i + 1]))
            if len(slice_arr) < min_slice_samples:
                skipped.append({"symbol": symbol, "reason": "too short"})
                continue
            normalised = audio.peak_normalise(slice_arr)

            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setnchannels(audio.CHANNELS)
                w.setsampwidth(audio.SAMPLE_WIDTH)
                w.setframerate(audio.SAMPLE_RATE)
                w.writeframes(normalised.tobytes())
            inventory.save_phoneme(symbol, out.getvalue())
            saved.append(symbol)

        log.info("extract/save: word=%r saved=%s skipped=%s", word, saved, skipped)
        return jsonify({"saved": saved, "skipped": skipped}), 200
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: `37 passed`.

- [ ] **Step 5: Confirm full suite**

Run: `uv run pytest -v`
Expected: 111 + 8 = `119 passed`.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Add POST /api/extract/save"
```

---

## Task 5: Editor CSS + JS scaffold (waveform + phoneme rows)

**Files:**
- Modify: `static/style.css`
- Modify: `static/app.js`

No automated tests for the frontend — manual verification in Task 8.

- [ ] **Step 1: Append editor styles to `static/style.css`**

```css
#phoneme-editor {
  background: var(--panel);
  border: 1px solid var(--accent);
  border-radius: var(--radius);
  margin: 1rem 0;
  padding: 1rem;
}

#phoneme-editor .editor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

#phoneme-editor .editor-header h3 {
  margin: 0;
  font-size: 1rem;
}

#phoneme-editor .dismiss {
  background: transparent;
  color: var(--muted);
  border: none;
  font-size: 1.3rem;
  cursor: pointer;
  padding: 0 0.4rem;
}

#phoneme-editor canvas {
  display: block;
  width: 100%;
  height: 120px;
  background: #1a1b1e;
  border-radius: 4px;
  cursor: ew-resize;
  margin-bottom: 0.6rem;
}

#phoneme-editor .row {
  display: grid;
  grid-template-columns: 1.5rem 2.5rem 4rem 2rem 1fr;
  gap: 0.6rem;
  align-items: center;
  padding: 0.25rem 0;
  font-family: ui-monospace, "Cascadia Mono", monospace;
}

#phoneme-editor .row .ex {
  font-family: system-ui, sans-serif;
  font-style: italic;
  color: var(--muted);
}

#phoneme-editor .row .duration {
  color: var(--muted);
}

#phoneme-editor .row .note {
  color: var(--muted);
  font-size: 0.85em;
  justify-self: end;
}

#phoneme-editor .row .play {
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
  border-radius: 4px;
  cursor: pointer;
  padding: 0.1rem 0.4rem;
}

#phoneme-editor .actions {
  margin-top: 0.8rem;
  display: flex;
  gap: 0.4rem;
}

#phoneme-editor .actions .cancel {
  background: #3a3d44;
  color: var(--fg);
}
```

- [ ] **Step 2: Add the editor element reference and module state at the top of `static/app.js`**

Find the `const els = { … }` block. The editor container is created dynamically, but add a reference for the host element where it will be inserted. Add this after the existing `phonemeGrid` entry:

```javascript
  recordPane: document.getElementById('record-pane'),
```

Add the module-level state (right after the existing `let currentPhoneme = null;` line):

```javascript
// Phoneme editor state — null when closed.
let editor = null;
```

- [ ] **Step 3: Add the editor scaffold functions at the END of `static/app.js`**

Append:

```javascript
async function openExtractEditor(word) {
  closeExtractEditor();

  let prop;
  try {
    const r = await fetch('/api/extract/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus(`Extract: ${err.error || r.statusText}`, true);
      return;
    }
    prop = await r.json();
  } catch (e) {
    setStatus(`Extract failed: ${e.message}`, true);
    return;
  }

  editor = {
    word: prop.word,
    audioBase64: prop.audio,
    audioDurationMs: prop.audio_duration_ms,
    phonemes: prop.phonemes,
    boundariesMs: prop.boundaries_ms.slice(),
    audioBuffer: null,
    audioCtx: null,
    container: null,
    canvas: null,
    samples: null,        // Int16Array, lazily decoded for waveform draw
    draggingI: null,
  };

  buildEditorDom();
  decodeWaveform();
}

function closeExtractEditor() {
  if (!editor) return;
  if (editor.container && editor.container.parentNode) {
    editor.container.parentNode.removeChild(editor.container);
  }
  editor = null;
}

function buildEditorDom() {
  const container = document.createElement('div');
  container.id = 'phoneme-editor';

  const header = document.createElement('div');
  header.className = 'editor-header';
  const title = document.createElement('h3');
  title.textContent = `Extracting phonemes from "${editor.word}"`;
  const dismiss = document.createElement('button');
  dismiss.className = 'dismiss';
  dismiss.textContent = '×';
  dismiss.addEventListener('click', closeExtractEditor);
  header.appendChild(title);
  header.appendChild(dismiss);
  container.appendChild(header);

  const canvas = document.createElement('canvas');
  canvas.width = 800;
  canvas.height = 120;
  container.appendChild(canvas);
  editor.canvas = canvas;

  for (let i = 0; i < editor.phonemes.length; i++) {
    const ph = editor.phonemes[i];
    const row = document.createElement('div');
    row.className = 'row';
    row.dataset.index = String(i);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !ph.recorded;
    checkbox.dataset.index = String(i);

    const sym = document.createElement('span');
    sym.className = 'symbol';
    sym.textContent = ph.symbol;

    const ex = document.createElement('span');
    ex.className = 'ex';
    ex.textContent = ph.example;

    const play = document.createElement('button');
    play.className = 'play';
    play.textContent = '▶';
    play.title = `Play ${ph.symbol}`;

    const duration = document.createElement('span');
    duration.className = 'duration';
    duration.textContent = `${editor.boundariesMs[i + 1] - editor.boundariesMs[i]} ms`;

    row.appendChild(checkbox);
    row.appendChild(sym);
    row.appendChild(ex);
    row.appendChild(play);
    row.appendChild(duration);

    if (ph.recorded) {
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = 'already recorded';
      row.appendChild(note);
    }

    container.appendChild(row);
  }

  const actions = document.createElement('div');
  actions.className = 'actions';
  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save phonemes';
  saveBtn.className = 'save';
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.className = 'cancel';
  cancelBtn.addEventListener('click', closeExtractEditor);
  actions.appendChild(saveBtn);
  actions.appendChild(cancelBtn);
  container.appendChild(actions);

  // Insert after the record pane.
  els.recordPane.insertAdjacentElement('afterend', container);
  editor.container = container;
}

async function decodeWaveform() {
  if (!editor) return;
  const bin = atob(editor.audioBase64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // Parse the WAV header manually to get raw int16 samples (faster than AudioContext for redraw).
  // Header is 44 bytes; samples are little-endian int16 starting at offset 44.
  const samplesArr = new Int16Array(bytes.buffer, 44, (bytes.length - 44) / 2);
  editor.samples = samplesArr;
  drawWaveform();
}

function drawWaveform() {
  if (!editor || !editor.canvas || !editor.samples) return;
  const canvas = editor.canvas;
  const ctx = canvas.getContext('2d');
  // Resize to actual pixel width (the CSS width was set to 100%; read it).
  const cssW = canvas.clientWidth;
  if (cssW > 0 && canvas.width !== cssW) {
    canvas.width = cssW;
  }
  const w = canvas.width;
  const h = canvas.height;
  const mid = h / 2;

  ctx.fillStyle = '#1a1b1e';
  ctx.fillRect(0, 0, w, h);

  const samples = editor.samples;
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const threshold = peak * 0.02;
  const bucketSize = samples.length / w;

  for (let x = 0; x < w; x++) {
    const start = Math.floor(x * bucketSize);
    const end = Math.floor((x + 1) * bucketSize);
    let bucketPeak = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(samples[i]);
      if (a > bucketPeak) bucketPeak = a;
    }
    const y = (bucketPeak / 32768) * (h / 2);
    const isLoud = bucketPeak >= threshold;
    ctx.fillStyle = isLoud ? '#8fdf9b' : '#4a4d54';
    ctx.fillRect(x, mid - y, 1, Math.max(1, y * 2));
  }

  // Draw boundary lines.
  ctx.strokeStyle = '#4ea8de';
  ctx.lineWidth = 2;
  for (const b of editor.boundariesMs) {
    const x = (b / editor.audioDurationMs) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}
```

- [ ] **Step 4: Wire a temporary smoke test — call `openExtractEditor("hello")` from the dev console**

Run (PowerShell):
```powershell
uv run flask --app app run
```

Open `http://127.0.0.1:5000`. Record the word "hello" if it isn't already saved. In the browser DevTools console, run:

```javascript
openExtractEditor("hello")
```

Expected:
- An editor panel appears below the record pane.
- It contains a waveform canvas with green (loud) and grey (silent) regions, vertical blue boundary lines.
- One row per phoneme of "hello" (HH AH L OW), each with a checkbox, symbol, example word, ▶ button, and live duration in ms.
- Save phonemes and Cancel buttons.
- × dismisses the editor.

The Save and ▶ buttons don't do anything yet — they're wired in Tasks 6 and 7.

Stop the server.

- [ ] **Step 5: Confirm test suite still passes**

Run: `uv run pytest -v`
Expected: 119 passed (no change — the JS is untested by pytest).

- [ ] **Step 6: Commit**

```bash
git add static/style.css static/app.js
git commit -m "Add phoneme extraction editor scaffold (canvas + rows)"
```

---

## Task 6: Editor boundary drag interaction

**Files:**
- Modify: `static/app.js`

- [ ] **Step 1: Add drag handling to `static/app.js`**

Find the `drawWaveform` function (added in Task 5). Right after it, append:

```javascript
function attachCanvasDrag() {
  const canvas = editor.canvas;
  const HANDLE_PX = 8;   // hit-test radius around a boundary

  canvas.addEventListener('mousedown', (e) => {
    if (!editor) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ms = (x / canvas.width) * editor.audioDurationMs;
    // Find nearest boundary.
    let nearestI = -1;
    let nearestPx = Infinity;
    for (let i = 0; i < editor.boundariesMs.length; i++) {
      const bx = (editor.boundariesMs[i] / editor.audioDurationMs) * canvas.width;
      const dist = Math.abs(bx - x);
      if (dist < nearestPx) {
        nearestPx = dist;
        nearestI = i;
      }
    }
    if (nearestI >= 0 && nearestPx <= HANDLE_PX) {
      editor.draggingI = nearestI;
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!editor || editor.draggingI === null) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    let ms = (x / canvas.width) * editor.audioDurationMs;
    const i = editor.draggingI;
    const minMs = i > 0 ? editor.boundariesMs[i - 1] + 5 : 0;
    const maxMs = i < editor.boundariesMs.length - 1
      ? editor.boundariesMs[i + 1] - 5
      : editor.audioDurationMs;
    ms = Math.max(minMs, Math.min(maxMs, ms));
    editor.boundariesMs[i] = Math.round(ms);
    drawWaveform();
    updateRowDurations();
  });

  window.addEventListener('mouseup', () => {
    if (editor) editor.draggingI = null;
  });
}

function updateRowDurations() {
  if (!editor || !editor.container) return;
  const rows = editor.container.querySelectorAll('.row');
  for (const row of rows) {
    const i = parseInt(row.dataset.index, 10);
    const dur = editor.boundariesMs[i + 1] - editor.boundariesMs[i];
    const durEl = row.querySelector('.duration');
    if (durEl) durEl.textContent = `${dur} ms`;
  }
}
```

- [ ] **Step 2: Call `attachCanvasDrag()` after building the DOM**

Find the `buildEditorDom` function. At the end of it (right after `editor.container = container;`), add:

```javascript
  attachCanvasDrag();
```

- [ ] **Step 3: Manual verification — drag boundaries**

Run: `uv run flask --app app run`

Open the app. Record "hello" if needed. Open the editor with:

```javascript
openExtractEditor("hello")
```

Verify:
- Hovering over a vertical blue boundary line shows the `ew-resize` cursor (from the canvas CSS).
- Click-and-drag any internal boundary horizontally. The line follows the cursor. The phoneme rows' duration values update in real time.
- The first and last boundaries are also draggable; the first cannot go below 0 ms, the last cannot exceed the audio duration.
- A boundary cannot cross its neighbour — it clamps 5 ms away.

Stop the server.

- [ ] **Step 4: Run the test suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: 119 passed.

- [ ] **Step 5: Commit**

```bash
git add static/app.js
git commit -m "Editor: draggable boundary handles update durations live"
```

---

## Task 7: Per-phoneme play, Save flow, and trigger wiring

**Files:**
- Modify: `static/app.js`

- [ ] **Step 1: Add audio decode + play helpers**

Append to `static/app.js`:

```javascript
async function ensureEditorAudioBuffer() {
  if (!editor) return null;
  if (editor.audioBuffer) return editor.audioBuffer;
  const bin = atob(editor.audioBase64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
  editor.audioCtx = ctx;
  // decodeAudioData consumes the ArrayBuffer; pass a slice to avoid breaking samples references.
  editor.audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
  return editor.audioBuffer;
}

async function playSlice(startMs, endMs) {
  const buffer = await ensureEditorAudioBuffer();
  if (!buffer) return;
  const src = editor.audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(editor.audioCtx.destination);
  const startSec = startMs / 1000;
  const durationSec = Math.max(0.01, (endMs - startMs) / 1000);
  src.start(0, startSec, durationSec);
}
```

- [ ] **Step 2: Wire the per-row ▶ buttons**

Find the section inside `buildEditorDom` where the row's `play` button is created (`play.title = ...`). REPLACE that part by adding a click listener. Locate this block:

```javascript
    const play = document.createElement('button');
    play.className = 'play';
    play.textContent = '▶';
    play.title = `Play ${ph.symbol}`;
```

and replace it with:

```javascript
    const play = document.createElement('button');
    play.className = 'play';
    play.textContent = '▶';
    play.title = `Play ${ph.symbol}`;
    play.addEventListener('click', (e) => {
      e.preventDefault();
      playSlice(editor.boundariesMs[i], editor.boundariesMs[i + 1]);
    });
```

- [ ] **Step 3: Wire the Save phonemes button**

Find the section inside `buildEditorDom` that creates `saveBtn`. Right after `saveBtn.className = 'save';`, add:

```javascript
    saveBtn.addEventListener('click', () => savePhonemesFromEditor());
```

Then append a new function at the end of `static/app.js`:

```javascript
async function savePhonemesFromEditor() {
  if (!editor) return;
  const word = editor.word;
  const boundariesMs = editor.boundariesMs.slice();
  const checkboxes = editor.container.querySelectorAll('.row input[type=checkbox]');
  const saveFlags = Array.from(checkboxes).map((cb) => cb.checked);

  let r;
  try {
    r = await fetch('/api/extract/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word, boundaries_ms: boundariesMs, save: saveFlags }),
    });
  } catch (e) {
    setStatus(`Save phonemes failed: ${e.message}`, true);
    return;
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`Save phonemes: ${err.error || r.statusText}`, true);
    return;
  }
  const { saved, skipped } = await r.json();
  let msg = `Saved phonemes: ${saved.length ? saved.join(', ') : 'none'}`;
  if (skipped.length) {
    const skipStr = skipped.map((s) => `${s.symbol} (${s.reason})`).join(', ');
    msg += ` — skipped: ${skipStr}`;
  }
  setStatus(msg);
  closeExtractEditor();
  await refreshInventory();
}
```

- [ ] **Step 4: Auto-trigger the editor after a successful word save**

Find the existing `saveWord` function. At the very end of its body (after `els.wordInput.value = '';`), add:

```javascript
  // After a successful word save, automatically offer phoneme extraction.
  openExtractEditor(saved);
```

So the full function body now reads (the change is the new last line):

```javascript
async function saveWord(word, blob) {
  const form = new FormData();
  form.append('word', word);
  form.append('audio', blob, 'audio.wav');

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
  els.wordInput.value = '';

  // After a successful word save, automatically offer phoneme extraction.
  openExtractEditor(saved);
}
```

- [ ] **Step 5: Add ✎ buttons to word chips**

Find `refreshWordInventory` (added in phase 2). REPLACE its body (keeping the function name) with:

```javascript
async function refreshWordInventory() {
  const r = await fetch('/api/inventory');
  const { words } = await r.json();
  els.inventoryChips.innerHTML = '';
  for (const w of words) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const label = document.createElement('span');
    label.textContent = w;
    chip.appendChild(label);

    const edit = document.createElement('button');
    edit.className = 'chip-edit';
    edit.textContent = '✎';
    edit.title = `Extract phonemes from "${w}"`;
    edit.addEventListener('click', (e) => {
      e.stopPropagation();
      openExtractEditor(w);
    });
    chip.appendChild(edit);

    els.inventoryChips.appendChild(chip);
  }
}
```

- [ ] **Step 6: Add a small style for the ✎ button**

Append to `static/style.css`:

```css
.chip-edit {
  background: transparent;
  border: none;
  color: var(--muted);
  cursor: pointer;
  margin-left: 0.3rem;
  padding: 0 0.2rem;
  font-size: 0.85em;
}
.chip-edit:hover { color: var(--accent); }
```

- [ ] **Step 7: Manual verification — end-to-end extraction**

Run: `uv run flask --app app run`

Open the app. Then:

1. Record a fresh word like "world".
2. After clicking Save, the extraction editor opens automatically below the record pane.
3. Click each phoneme's ▶ — you hear the proposed slice for that phoneme.
4. Drag a boundary. The duration in that row updates live, and the play preview now reflects the new range.
5. For phonemes that already exist in the phoneme inventory, the checkbox is unchecked by default. Confirm by inspecting an existing-phoneme row.
6. Click "Save phonemes". Status line shows what was saved and what (if anything) was skipped. The editor dismisses. The phoneme grid below refreshes with the newly-recorded chips turned green.
7. Click the ✎ button on a word chip in the inventory bar. The editor opens again for that word.
8. Try an out-of-dict word: record "xkcdfoo" — after save, no editor opens (the status line shows the OOV message instead).

Stop the server.

- [ ] **Step 8: Run the full test suite**

Run: `uv run pytest -v`
Expected: 119 passed.

- [ ] **Step 9: Commit**

```bash
git add static/app.js static/style.css
git commit -m "Editor: per-phoneme play, save flow, auto-trigger and chip pencil"
```

---

## Task 8: README + acceptance run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`**

Open `README.md`. Find the `## Phoneme fallback` section (added in phase 2). After it, insert a new section:

```markdown
## Phoneme extraction

Recording a word also offers to chop it into per-phoneme slices. After you save
a word, an editor appears showing the waveform with auto-proposed phoneme
boundaries (duration-weighted from CMUdict's phoneme sequence). Drag the
boundary lines to align them with what you actually said, click ▶ to preview
each slice, untick any you don't want, then click *Save phonemes* — the
selected slices are written to the phoneme inventory.

The proposed boundaries are crude (linguistic priors only — no acoustic
alignment). Expect to drag every boundary by hand for usable phoneme audio.
The ✎ icon on each word chip re-opens the editor later.
```

- [ ] **Step 2: Acceptance run**

Run: `uv run pytest -v`
Expected: all tests pass (target: 119).

Run: `uv run flask --app app run`

Full manual workflow:

1. Open `http://127.0.0.1:5000`.
2. Record a word ("hello"). Confirm the editor auto-opens.
3. Inside the editor, drag boundaries and play each phoneme to find good cuts.
4. Click *Save phonemes*. Confirm the phoneme grid below refreshes with new green chips.
5. Synthesise a sentence containing a word whose phonemes you just extracted but didn't record directly. Confirm playback succeeds via the phoneme path.
6. Click ✎ on an existing word chip; confirm the editor opens for re-extraction.
7. Try an out-of-dictionary word — confirm the editor does NOT open and the status line says so.

Stop the server.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document phoneme extraction in README"
```

---

## Self-review

**Spec coverage:**

- ✅ `extraction.py` with duration table + `propose_boundaries` (loud-region trim, prior-weighted allocation) — Task 1.
- ✅ `audio.slice_samples` helper — Task 2.
- ✅ `POST /api/extract/propose` returning audio + boundaries + phoneme metadata — Task 3.
- ✅ `POST /api/extract/save` with all validation paths, 30 ms slice floor, overwrite semantics — Task 4.
- ✅ Editor scaffold (canvas + per-phoneme rows) — Task 5.
- ✅ Draggable boundary handles with neighbour clamping and live duration updates — Task 6.
- ✅ Per-phoneme play, Save phonemes flow, auto-trigger after word save, chip ✎ button — Task 7.
- ✅ README + acceptance — Task 8.
- ✅ All listed test cases distributed across Tasks 1-4.

**Placeholder scan:** No TBDs, no vague "handle errors" steps. The "Expected: 119 passed" count is approximate and may differ by a few tests if prior tasks accumulate test counts slightly differently — that's a non-issue.

**Type consistency:**
- `propose_boundaries(samples, phoneme_seq, sample_rate=SAMPLE_RATE) -> list[int]` matches usage in `app.py`.
- `slice_samples(samples, start_ms, end_ms, sample_rate=SAMPLE_RATE) -> np.ndarray` matches usage in the save route.
- `boundaries_ms` is a `list[int]` everywhere (sent as JSON list; coerced to int when slicing).
- `save` is `list[bool]` of length `N` for `N` phonemes.
- API response shapes match the spec.

**Plan complete.**

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-21-phoneme-extraction.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans.

Which approach?
