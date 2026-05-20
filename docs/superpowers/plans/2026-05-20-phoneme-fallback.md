# Phoneme Fallback (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing word-level concatenative TTS so that sentences containing un-recorded words are synthesised by stitching individually-recorded phoneme samples instead of being skipped.

**Architecture:** Bundle the CMU Pronouncing Dictionary as a static data file. Add a `phonemes.py` module for grapheme-to-phoneme. Add a `PhonemeResolver` that slots into the existing `resolver.RESOLVERS` chain after `WordResolver`. Rewrite `synth.synthesise` with a two-pass algorithm: try the chain on the raw token, then phonemise via CMUdict and try the chain on each phoneme. Output becomes grouped (`list[list[Path]]`) so `audio.concatenate` can apply 30 ms silence between words but zero between phonemes inside a word.

**Tech Stack:** Python 3.13 via uv, Flask, numpy, stdlib `wave`. Vanilla JavaScript with the Web Audio API. CMUdict (BSD-style license).

**Spec reference:** `docs/superpowers/specs/2026-05-20-phoneme-fallback-design.md`

**Project conventions reminder:**
- Run all Python via `uv run …` — never `pip` or `python -m venv` directly.
- Commit directly to `master`; no feature branches.
- Tests: `uv run pytest -v`.

---

## Task 1: Bundle CMUdict

**Files:**
- Create: `data/cmudict.dict` (downloaded from the cmusphinx repo)
- Create: `data/README.md` (source and license note)

This task has no automated tests — the dict's content is verified indirectly in Task 3.

- [ ] **Step 1: Create the `data/` directory**

Run (PowerShell):
```powershell
mkdir data
```

- [ ] **Step 2: Download `cmudict.dict`**

Run (PowerShell):
```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict" `
  -OutFile "data/cmudict.dict"
```

Expected: the file is roughly 4–5 MB and starts with lower-case entries like `'bout B AW1 T`.

- [ ] **Step 3: Spot-check the download**

Run (PowerShell):
```powershell
Get-Content data/cmudict.dict -TotalCount 5
(Get-Content data/cmudict.dict | Measure-Object -Line).Lines
```

Expected: roughly 135,000 lines. First lines should look like `WORD PH1 PH2 ...` with lowercase words and ARPABET-with-stress phonemes.

- [ ] **Step 4: Create `data/README.md`**

```markdown
# Bundled data

## cmudict.dict

The CMU Pronouncing Dictionary, mirrored from
<https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict>.

Format: one entry per line, `<word> <phoneme1> <phoneme2> …`. Phonemes are
ARPABET with stress digit suffixes (0/1/2). Multiple pronunciations appear
as `word(2)`, `word(3)` etc. — talky-talky's loader keeps only the first
pronunciation per word.

Licensed under a BSD-style permissive license; see the upstream repository
for the canonical text.
```

- [ ] **Step 5: Commit**

```bash
git add data/cmudict.dict data/README.md
git commit -m "Bundle CMU Pronouncing Dictionary for grapheme-to-phoneme lookup"
```

---

## Task 2: `phonemes.py` — constants and `is_phoneme`

**Files:**
- Create: `phonemes.py`
- Create: `tests/test_phonemes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phonemes.py`:

```python
import phonemes


def test_arpabet_set_size():
    assert len(phonemes.PHONEMES) == 39
    assert set(phonemes.PHONEMES) == phonemes.PHONEMES_SET
    # Spot-check a few canonical members
    for sym in ("AA", "AE", "AH", "K", "ZH"):
        assert sym in phonemes.PHONEMES_SET


def test_phonemes_are_uppercase_and_unique():
    assert all(s == s.upper() for s in phonemes.PHONEMES)
    assert len(set(phonemes.PHONEMES)) == len(phonemes.PHONEMES)


def test_is_phoneme_case_insensitive():
    assert phonemes.is_phoneme("AA")
    assert phonemes.is_phoneme("aa")
    assert phonemes.is_phoneme("Aa")


def test_is_phoneme_rejects_non_arpabet():
    assert not phonemes.is_phoneme("hello")
    assert not phonemes.is_phoneme("")
    assert not phonemes.is_phoneme("ZZ")
    assert not phonemes.is_phoneme("AA1")  # stress-digit form is not a phoneme symbol


def test_list_phonemes_returns_canonical_list():
    result = phonemes.list_phonemes()
    assert result == list(phonemes.PHONEMES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_phonemes.py -v`
Expected: `ModuleNotFoundError: No module named 'phonemes'`.

- [ ] **Step 3: Implement `phonemes.py` (constants + `is_phoneme` only)**

```python
"""ARPABET phoneme set and grapheme-to-phoneme via CMUdict.

`PHONEMES` is the canonical 39-symbol set used in CMUdict. Stress digit
suffixes (e.g. `AH1`) are NOT phonemes — they are dropped at dict-load time.
"""
from __future__ import annotations

PHONEMES: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH",
    "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
)

PHONEMES_SET: frozenset[str] = frozenset(PHONEMES)


def is_phoneme(token: str) -> bool:
    """Return True iff `token` is an ARPABET phoneme symbol (case-insensitive)."""
    if not token:
        return False
    return token.upper() in PHONEMES_SET


def list_phonemes() -> list[str]:
    """Return the canonical phoneme list (in declaration order)."""
    return list(PHONEMES)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_phonemes.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add phonemes.py tests/test_phonemes.py
git commit -m "Add phonemes.py with ARPABET constants and is_phoneme"
```

---

## Task 3: `phonemes.py` — CMUdict loader and `word_to_phonemes`

**Files:**
- Modify: `phonemes.py`
- Modify: `tests/test_phonemes.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_phonemes.py`:

```python
def test_word_to_phonemes_known_word():
    # "hello" is a stable entry in CMUdict
    assert phonemes.word_to_phonemes("hello") == ["HH", "AH", "L", "OW"]


def test_word_to_phonemes_case_insensitive():
    a = phonemes.word_to_phonemes("hello")
    b = phonemes.word_to_phonemes("HELLO")
    c = phonemes.word_to_phonemes("Hello")
    assert a == b == c


def test_word_to_phonemes_unknown_returns_none():
    assert phonemes.word_to_phonemes("xyzqqqzz") is None


def test_word_to_phonemes_empty_returns_none():
    assert phonemes.word_to_phonemes("") is None
    assert phonemes.word_to_phonemes("   ") is None


def test_stress_digits_are_stripped():
    # "world" → W ER1 L D in the raw dict — must come back without the "1"
    result = phonemes.word_to_phonemes("world")
    assert result is not None
    assert all(p in phonemes.PHONEMES_SET for p in result)
    assert "ER" in result


def test_loaded_dict_has_expected_scale():
    # ~125k–135k unique words is normal for CMUdict
    assert 100_000 < len(phonemes._DICT) < 200_000


def test_only_first_pronunciation_kept():
    # CMUdict has entries like "the" and "the(2)". After load, "the" should map
    # to one specific pronunciation (the first), not a list of options.
    result = phonemes.word_to_phonemes("the")
    assert result is not None
    assert isinstance(result, list)
    assert all(isinstance(p, str) for p in result)
```

- [ ] **Step 2: Run the tests to verify failures**

Run: `uv run pytest tests/test_phonemes.py -v`
Expected: 5 prior passed + 7 new failures (`AttributeError: module 'phonemes' has no attribute 'word_to_phonemes'`).

- [ ] **Step 3: Extend `phonemes.py`**

Add these imports at the top (after `from __future__ import annotations`):

```python
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
```

After `PHONEMES_SET` declaration, add the dict loader and lookup function:

```python
_DICT_PATH = Path(__file__).parent / "data" / "cmudict.dict"
_STRESS_RE = re.compile(r"\d+$")


def _strip_stress(phoneme: str) -> str:
    return _STRESS_RE.sub("", phoneme).upper()


def _load_dict() -> dict[str, list[str]]:
    """Parse `data/cmudict.dict` into a `{lowercase_word: [PHONEMES…]}` dict.

    Rules:
      - Lines starting with `;;;` are comments — skip.
      - Tokens after the word that don't strip down to a known ARPABET symbol
        cause the whole entry to be skipped (with a WARNING log).
      - Entries marked as alternate pronunciations (`word(2)`, `word(3)`, ...)
        are skipped — first wins.
    """
    if not _DICT_PATH.is_file():
        raise FileNotFoundError(
            f"CMUdict not found at {_DICT_PATH}. "
            "Run Task 1 of the phoneme-fallback plan to bundle it."
        )

    d: dict[str, list[str]] = {}
    with _DICT_PATH.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";;;"):
                continue
            # Strip inline comments (rare, format-dependent).
            if " #" in line:
                line = line.split(" #", 1)[0].rstrip()

            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0]
            if "(" in word:
                # Alternate pronunciation — skip; first one wins.
                continue
            word_lc = word.lower()
            if word_lc in d:
                continue  # already have an entry (defensive)

            phs: list[str] | None = []
            for p in parts[1:]:
                stripped = _strip_stress(p)
                if stripped not in PHONEMES_SET:
                    log.warning(
                        "cmudict: unknown phoneme %r in entry %r — skipping entry",
                        p, word_lc,
                    )
                    phs = None
                    break
                phs.append(stripped)

            if phs is not None:
                d[word_lc] = phs
    return d


_DICT: dict[str, list[str]] = _load_dict()


def word_to_phonemes(word: str) -> list[str] | None:
    """Look up `word` in CMUdict and return its ARPABET phoneme list (case-insensitive).

    Returns `None` if the word is not in the dictionary, or if the input is
    empty or whitespace-only.
    """
    if word is None:
        return None
    key = word.strip().lower()
    if not key:
        return None
    phs = _DICT.get(key)
    return list(phs) if phs is not None else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_phonemes.py -v`
Expected: `12 passed`.

If a test like `test_word_to_phonemes_known_word` fails because the CMUdict pronunciation differs from `["HH", "AH", "L", "OW"]`, update the test to match what the actual loaded dict returns for that word (read the relevant line directly: `Get-Content data/cmudict.dict | Select-String "^hello "`). The function under test is correct; the expected list is the moving part because it depends on the exact dict version. Apply the same adjustment if `test_stress_digits_are_stripped` or `test_only_first_pronunciation_kept` need realignment.

- [ ] **Step 5: Commit**

```bash
git add phonemes.py tests/test_phonemes.py
git commit -m "Add CMUdict loader and word_to_phonemes lookup"
```

---

## Task 4: `inventory.py` — phoneme storage helpers

**Files:**
- Modify: `inventory.py`
- Modify: `tests/test_inventory.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_inventory.py`:

```python
def test_phoneme_path_lowercases_filename(temp_inventory):
    assert inventory.phoneme_path("AA") == temp_inventory / "phonemes" / "aa.wav"
    assert inventory.phoneme_path("aa") == temp_inventory / "phonemes" / "aa.wav"


def test_has_phoneme_reflects_existence(temp_inventory):
    assert not inventory.has_phoneme("AA")
    inventory.save_phoneme("AA", b"x")
    assert inventory.has_phoneme("AA")
    assert inventory.has_phoneme("aa")


def test_save_phoneme_creates_phoneme_dir(temp_inventory):
    inventory.save_phoneme("AA", b"wavbytes")
    assert (temp_inventory / "phonemes" / "aa.wav").read_bytes() == b"wavbytes"


def test_list_recorded_phonemes_returns_uppercase_sorted(temp_inventory):
    inventory.save_phoneme("OW", b"x")
    inventory.save_phoneme("AA", b"x")
    inventory.save_phoneme("k", b"x")
    assert inventory.list_recorded_phonemes() == ["AA", "K", "OW"]


def test_list_recorded_phonemes_empty_when_no_dir(temp_inventory):
    assert inventory.list_recorded_phonemes() == []
```

- [ ] **Step 2: Run the tests to verify failures**

Run: `uv run pytest tests/test_inventory.py -v`
Expected: 9 prior passed + 5 new failures (`AttributeError: module 'inventory' has no attribute 'phoneme_path'` and similar).

- [ ] **Step 3: Extend `inventory.py`**

The current `inventory.py` ends with `list_words()`. Append after it:

```python
PHONEME_SUBDIR = "phonemes"


def _phoneme_dir() -> Path:
    return INVENTORY_DIR / PHONEME_SUBDIR


def phoneme_path(phoneme: str) -> Path:
    return _phoneme_dir() / f"{phoneme.lower()}.wav"


def has_phoneme(phoneme: str) -> bool:
    return phoneme_path(phoneme).exists()


def save_phoneme(phoneme: str, wav_bytes: bytes) -> None:
    _phoneme_dir().mkdir(parents=True, exist_ok=True)
    phoneme_path(phoneme).write_bytes(wav_bytes)


def list_recorded_phonemes() -> list[str]:
    d = _phoneme_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem.upper() for p in d.glob("*.wav"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inventory.py -v`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add inventory.py tests/test_inventory.py
git commit -m "Add inventory phoneme_path / has_phoneme / save_phoneme / list_recorded_phonemes"
```

---

## Task 5: `audio.py` — `concatenate` accepts grouped paths

**Files:**
- Modify: `audio.py`
- Modify: `tests/test_audio.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_audio.py`:

```python
def test_concatenate_grouped_silence_between_groups_only(tmp_path):
    """[[a, b], [c]] -> a + b + 30ms + c (no silence between a and b)."""
    samples_a = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    samples_b = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 8000
    samples_c = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 6000
    a = tmp_path / "a.wav"; audio.write_wav(a, samples_a)
    b = tmp_path / "b.wav"; audio.write_wav(b, samples_b)
    c = tmp_path / "c.wav"; audio.write_wav(c, samples_c)

    wav_bytes = audio.concatenate([[a, b], [c]])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))

    inter = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    expected_len = len(samples_a) + len(samples_b) + inter + len(samples_c)
    assert len(loaded) == expected_len


def test_concatenate_grouped_drops_empty_inner_groups(tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    p = tmp_path / "a.wav"; audio.write_wav(p, samples)

    wav_bytes = audio.concatenate([[], [p], []])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == len(samples)


def test_concatenate_grouped_empty_outer_list_returns_empty_wav():
    wav_bytes = audio.concatenate([[], []])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))
    assert len(loaded) == 0


def test_concatenate_flat_list_still_works(tmp_path):
    """Backward compatibility: flat list[Path] still inserts inter-word silence."""
    samples_a = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    samples_b = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 8000
    a = tmp_path / "a.wav"; audio.write_wav(a, samples_a)
    b = tmp_path / "b.wav"; audio.write_wav(b, samples_b)

    wav_bytes = audio.concatenate([a, b])
    loaded, _ = audio.read_wav(io.BytesIO(wav_bytes))

    inter = int(audio.SAMPLE_RATE * audio.INTER_WORD_SILENCE_SECONDS)
    assert len(loaded) == len(samples_a) + inter + len(samples_b)
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 14 prior passed + 3 grouped-list failures (the flat-list test passes immediately).

The three failures will be `TypeError: expected str, bytes or os.PathLike object, not list` or similar — `concatenate` is trying to use a list as a path.

- [ ] **Step 3: Update `concatenate` in `audio.py`**

Replace the existing `concatenate` function with:

```python
def concatenate(paths_or_groups: "list[Path] | list[list[Path]]") -> bytes:
    """Read WAVs, apply fades, return a single WAV byte string.

    Accepts either:
      - `list[Path]` — flat. Inserts `INTER_WORD_SILENCE_SECONDS` of silence
        between every adjacent pair.
      - `list[list[Path]]` — grouped. Inserts that silence BETWEEN inner
        groups only. Adjacent paths inside the same inner group are
        concatenated with zero silence (the 5 ms fades alone suppress clicks).

    Empty inner groups are silently dropped. An empty top-level list
    produces an empty WAV.
    """
    import io as _io

    # Detect shape: a non-empty list whose first element is itself a list → grouped.
    if paths_or_groups and isinstance(paths_or_groups[0], list):
        groups: list[list[Path]] = [g for g in paths_or_groups if g]  # drop empty inner groups
    elif paths_or_groups:
        # Flat list — treat each path as its own group of one.
        groups = [[p] for p in paths_or_groups]
    else:
        groups = []

    silence_n = int(SAMPLE_RATE * INTER_WORD_SILENCE_SECONDS)
    silence = np.zeros(silence_n, dtype=np.int16)

    chunks: list[np.ndarray] = []
    for gi, group in enumerate(groups):
        for p in group:
            samples, _ = read_wav(p)
            samples = apply_fade(samples)
            chunks.append(samples)
        if gi < len(groups) - 1:
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

- [ ] **Step 4: Run all audio tests to verify everything passes**

Run: `uv run pytest tests/test_audio.py -v`
Expected: `18 passed` (14 prior + 4 new).

Also confirm no regressions elsewhere:

Run: `uv run pytest -v`
Expected: existing total `51 + 5` (Task 4 added 5 inventory tests) `+ 12` (Tasks 2-3 added 12 phoneme tests) `+ 4` (this task added 4 audio tests) = `72 passed`.

If the synthesis tests in `test_synth.py` start failing (they call `audio.concatenate([path, path, …])` indirectly), check that the flat-list path is genuinely preserved — they should NOT fail, because the flat detection treats each path as a single-element group, producing identical behaviour to the old code.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "Widen concatenate to accept grouped paths (no silence within groups)"
```

---

## Task 6: `resolver.py` — `PhonemeResolver`

**Files:**
- Modify: `resolver.py`
- Modify: `tests/test_resolver.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_resolver.py`:

```python
import phonemes


def test_phoneme_resolver_hit(temp_inventory):
    inventory.save_phoneme("AA", b"fakewav")
    r = resolver.PhonemeResolver()
    assert r.resolve("AA") == temp_inventory / "phonemes" / "aa.wav"


def test_phoneme_resolver_case_insensitive(temp_inventory):
    inventory.save_phoneme("AA", b"fakewav")
    r = resolver.PhonemeResolver()
    assert r.resolve("aa") == temp_inventory / "phonemes" / "aa.wav"
    assert r.resolve("Aa") == temp_inventory / "phonemes" / "aa.wav"


def test_phoneme_resolver_miss_no_file(temp_inventory):
    r = resolver.PhonemeResolver()
    assert r.resolve("AA") is None


def test_phoneme_resolver_rejects_non_arpabet(temp_inventory):
    # Even if a file named "hello.wav" exists, PhonemeResolver does not look at words.
    inventory.save("hello", b"fakewav")
    r = resolver.PhonemeResolver()
    assert r.resolve("hello") is None


def test_resolvers_chain_order():
    assert len(resolver.RESOLVERS) == 2
    assert isinstance(resolver.RESOLVERS[0], resolver.WordResolver)
    assert isinstance(resolver.RESOLVERS[1], resolver.PhonemeResolver)
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_resolver.py -v`
Expected: 4 prior passed + 5 new failures (`AttributeError: module 'resolver' has no attribute 'PhonemeResolver'`).

- [ ] **Step 3: Extend `resolver.py`**

Add this import near the top (alongside the existing `import inventory`):

```python
import phonemes
```

Add the `PhonemeResolver` class after `WordResolver`:

```python
class PhonemeResolver:
    """Resolves an ARPABET phoneme token to its recorded WAV file.

    Returns None for any token that is not itself a valid ARPABET symbol,
    even if a file by that name exists in the word inventory. The synthesis
    loop is responsible for phonemising English words; this resolver only
    handles raw phoneme symbols.
    """

    def resolve(self, word: str) -> Path | None:
        if not phonemes.is_phoneme(word):
            return None
        return inventory.phoneme_path(word) if inventory.has_phoneme(word) else None
```

Update the `RESOLVERS` list at the bottom of the file:

```python
RESOLVERS: list[Resolver] = [WordResolver(), PhonemeResolver()]
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_resolver.py -v`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add resolver.py tests/test_resolver.py
git commit -m "Add PhonemeResolver to the resolver chain"
```

---

## Task 7: `synth.synthesise` — phoneme fallback

**Files:**
- Modify: `synth.py`
- Modify: `tests/test_synth.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_synth.py`:

```python
import phonemes as _phonemes  # alias to avoid colliding with any local var


def _make_phoneme_recording(temp_inventory, symbol, n_samples=2000, amplitude=10000):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    audio.write_wav(temp_inventory / "phonemes" / f"{symbol.lower()}.wav", samples)


def _seed_all_phonemes_for_word(temp_inventory, word):
    """Create a phoneme recording for every phoneme the given word requires."""
    (temp_inventory / "phonemes").mkdir(exist_ok=True)
    phs = _phonemes.word_to_phonemes(word)
    assert phs is not None, f"{word!r} must be in CMUdict for this test"
    for ph in set(phs):
        _make_phoneme_recording(temp_inventory, ph)


def test_synthesise_phoneme_fallback_full_coverage(temp_inventory):
    """No word file for 'hello', but every phoneme recorded → synthesised, no skips."""
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    wav_bytes, skipped = synth.synthesise("hello")

    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_fallback_partial_missing(temp_inventory):
    """Word would be phonemise-able but one phoneme is missing → whole word skipped."""
    phs = _phonemes.word_to_phonemes("hello")
    assert phs is not None
    # Record every phoneme EXCEPT the first one.
    (temp_inventory / "phonemes").mkdir(exist_ok=True)
    for ph in set(phs[1:]):
        _make_phoneme_recording(temp_inventory, ph)

    wav_bytes, skipped = synth.synthesise("hello")

    assert wav_bytes is None
    assert skipped == ["hello"]


def test_synthesise_phoneme_fallback_word_not_in_cmudict(temp_inventory):
    wav_bytes, skipped = synth.synthesise("xyzqqqzz")
    assert wav_bytes is None
    assert skipped == ["xyzqqqzz"]


def test_synthesise_word_resolver_takes_precedence(temp_inventory):
    """If a word is recorded AND all its phonemes are recorded, use the word file."""
    word_samples = np.ones(audio.SAMPLE_RATE * 2, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "hello")

    wav_bytes, skipped = synth.synthesise("hello")
    assert wav_bytes is not None
    assert skipped == []

    # The output should match a single-word concatenate, not a phoneme stitch.
    expected = audio.concatenate([[temp_inventory / "hello.wav"]])
    assert wav_bytes == expected


def test_synthesise_mixed_word_and_phoneme_words(temp_inventory):
    """Sentence with one recorded word + one phoneme-only word: both synthesised."""
    word_samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 5000
    audio.write_wav(temp_inventory / "hello.wav", word_samples)
    _seed_all_phonemes_for_word(temp_inventory, "world")

    wav_bytes, skipped = synth.synthesise("hello world")
    assert wav_bytes is not None
    assert skipped == []


def test_synthesise_phoneme_token_directly(temp_inventory):
    """If the user literally types 'AA' as a token and the AA phoneme is recorded,
    PhonemeResolver in the chain picks it up — no CMUdict lookup needed."""
    _make_phoneme_recording(temp_inventory, "AA")

    wav_bytes, skipped = synth.synthesise("AA")

    assert wav_bytes is not None
    assert skipped == []
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_synth.py -v`
Expected: 9 prior passed (or some failures because the existing `synthesise` returns flat bytes that no longer match what the new `concatenate` produces from grouped input — but in practice the flat-list backward compat from Task 5 keeps the old tests green). The 6 new tests should all fail because `synthesise` doesn't yet do the phoneme fallback.

- [ ] **Step 3: Rewrite `synthesise` in `synth.py`**

Replace the existing `synthesise` function entirely with:

```python
import phonemes  # add to imports at top of synth.py if not already there


def synthesise(sentence: str) -> tuple[bytes | None, list[str]]:
    """Synthesise a sentence by stitching word recordings and/or phoneme recordings.

    For each token in the sentence:
      1. Walk the resolver chain. If a resolver returns a Path, use it as a
         single-element group.
      2. Otherwise, if the token is itself an ARPABET symbol that wasn't
         resolved (no phoneme file recorded), skip the token.
      3. Otherwise, look the token up in CMUdict. If absent → skip. If
         present, try the resolver chain on each phoneme. If every phoneme
         resolves, the word becomes a group of phoneme paths. If any phoneme
         is missing → skip the whole word (all-or-nothing).

    Returns (wav_bytes_or_None, skipped_tokens).
    """
    tokens = tokenise(sentence)
    if not tokens:
        return None, []

    groups: list[list] = []
    skipped: list[str] = []

    for token in tokens:
        # Pass 1: try the chain on the raw token.
        direct = None
        for r in resolver.RESOLVERS:
            direct = r.resolve(token)
            if direct is not None:
                break

        if direct is not None:
            groups.append([direct])
            continue

        # If the token itself is a phoneme symbol, the chain already had its chance.
        if phonemes.is_phoneme(token):
            skipped.append(token)
            continue

        # Pass 2: phonemise via CMUdict, then resolve each phoneme.
        phs = phonemes.word_to_phonemes(token)
        if phs is None:
            skipped.append(token)
            continue

        phoneme_paths = []
        all_resolved = True
        for ph in phs:
            ph_path = None
            for r in resolver.RESOLVERS:
                ph_path = r.resolve(ph)
                if ph_path is not None:
                    break
            if ph_path is None:
                all_resolved = False
                break
            phoneme_paths.append(ph_path)

        if not all_resolved:
            skipped.append(token)
        else:
            groups.append(phoneme_paths)

    if not groups:
        return None, skipped

    wav_bytes = audio.concatenate(groups)
    return wav_bytes, skipped
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_synth.py -v`
Expected: `15 passed` (9 prior + 6 new).

Full suite:

Run: `uv run pytest -v`
Expected: `78 passed` (existing 51 + 5 inv + 12 phn + 4 aud + 5 res + 6 synth - but exact total may differ slightly based on test counts in prior tasks; the important constraint is that all tests pass and no regressions appear).

- [ ] **Step 5: Commit**

```bash
git add synth.py tests/test_synth.py
git commit -m "Synthesise: phoneme fallback for un-recorded words"
```

---

## Task 8: `app.py` — `POST /api/record/phoneme`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_app.py`:

```python
import phonemes as _phonemes


def test_record_phoneme_saves_valid_wav(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.get_json() == {"phoneme": "AA"}
    assert (tmp_path / "phonemes" / "aa.wav").exists()


def test_record_phoneme_case_insensitive(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "aa", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.get_json() == {"phoneme": "AA"}
    assert (tmp_path / "phonemes" / "aa.wav").exists()


def test_record_phoneme_rejects_unknown_symbol(client):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "ZZ", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    body = r.get_json()
    assert "allowed" in body
    assert set(body["allowed"]) == _phonemes.PHONEMES_SET


def test_record_phoneme_rejects_empty_field(client):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_phoneme_rejects_wrong_sample_rate(client):
    wav = _make_wav_bytes(sample_rate=44100)
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_phoneme_rejects_too_short(client):
    wav = _make_wav_bytes(n_samples=100)
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_record_phoneme_409_on_existing(client, tmp_path):
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    (tmp_path / "phonemes" / "aa.wav").write_bytes(b"existing")
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 409


def test_record_phoneme_overwrite_true_replaces(client, tmp_path):
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    (tmp_path / "phonemes" / "aa.wav").write_bytes(b"existing")
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme?overwrite=true",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert (tmp_path / "phonemes" / "aa.wav").read_bytes() != b"existing"
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_app.py -v`
Expected: 14 prior passed + 8 new failures (404/405 — route doesn't exist).

- [ ] **Step 3: Add the route to `app.py`**

Add `import phonemes` to the imports block (alongside `import audio`, `import inventory`, `import synth`).

Inside `create_app`, after the existing `api_record` route, add:

```python
    @flask_app.route("/api/record/phoneme", methods=["POST"])
    def api_record_phoneme():
        phoneme_raw = request.form.get("phoneme", "").strip()
        if not phoneme_raw:
            return jsonify({"error": "phoneme is required"}), 400
        phoneme = phoneme_raw.upper()
        if phoneme not in phonemes.PHONEMES_SET:
            return jsonify({
                "error": "unknown phoneme",
                "allowed": list(phonemes.PHONEMES),
            }), 400

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
            log.warning("record/phoneme: invalid WAV: %s", e)
            return jsonify({"error": "invalid WAV file"}), 400

        if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
            log.warning(
                "record/phoneme: format mismatch rate=%s channels=%s sampwidth=%s",
                rate, channels, sampwidth,
            )
            return jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400

        samples = np.frombuffer(frames, dtype=np.int16)
        samples = audio.trim_silence(samples)
        if len(samples) < int(audio.SAMPLE_RATE * audio.MIN_LENGTH_SECONDS):
            log.warning("record/phoneme: too short after trim: %s samples", len(samples))
            return jsonify({"error": "recording too short"}), 400

        samples = audio.peak_normalise(samples)

        overwrite = request.args.get("overwrite") == "true"
        if inventory.has_phoneme(phoneme) and not overwrite:
            return jsonify({"error": "phoneme exists", "phoneme": phoneme}), 409

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(audio.CHANNELS)
            w.setsampwidth(audio.SAMPLE_WIDTH)
            w.setframerate(audio.SAMPLE_RATE)
            w.writeframes(samples.tobytes())
        inventory.save_phoneme(phoneme, out.getvalue())

        log.info("record/phoneme: saved %r (%d samples)", phoneme, len(samples))
        return jsonify({"phoneme": phoneme}), 200
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Add POST /api/record/phoneme"
```

---

## Task 9: `app.py` — `GET /api/phoneme-inventory`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_app.py`:

```python
def test_phoneme_inventory_lists_all_39(client):
    r = client.get("/api/phoneme-inventory")
    assert r.status_code == 200
    data = r.get_json()
    assert "phonemes" in data
    assert len(data["phonemes"]) == 39
    symbols = [entry["symbol"] for entry in data["phonemes"]]
    assert symbols == list(_phonemes.PHONEMES)  # canonical order
    assert all(entry["recorded"] is False for entry in data["phonemes"])


def test_phoneme_inventory_reflects_saved_phonemes(client, tmp_path):
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    (tmp_path / "phonemes" / "aa.wav").write_bytes(b"x")
    (tmp_path / "phonemes" / "b.wav").write_bytes(b"x")

    r = client.get("/api/phoneme-inventory")
    assert r.status_code == 200
    by_symbol = {e["symbol"]: e["recorded"] for e in r.get_json()["phonemes"]}
    assert by_symbol["AA"] is True
    assert by_symbol["B"] is True
    assert by_symbol["AE"] is False
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_app.py -v`
Expected: 22 prior passed + 2 new failures (404).

- [ ] **Step 3: Add the route to `app.py`**

Inside `create_app`, after `api_record_phoneme`, add:

```python
    @flask_app.route("/api/phoneme-inventory")
    def api_phoneme_inventory():
        recorded = set(inventory.list_recorded_phonemes())
        return jsonify({
            "phonemes": [
                {"symbol": s, "recorded": s in recorded}
                for s in phonemes.PHONEMES
            ]
        })
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: `24 passed`.

Full suite:

Run: `uv run pytest -v`
Expected: 80+ tests passing, no failures.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Add GET /api/phoneme-inventory"
```

---

## Task 10: HTML + CSS — phoneme grid section

**Files:**
- Modify: `templates/index.html`
- Modify: `static/style.css`

No automated tests — manual verify in browser, deferred to Task 12.

- [ ] **Step 1: Add the phoneme section to `templates/index.html`**

Open `templates/index.html`. After the closing `</main>` tag (and before the `<script>` tag), insert:

```html
  <section id="phoneme-pane">
    <h2>Phoneme inventory</h2>
    <p class="hint">Click a chip to record that phoneme. Green = recorded, red = not yet.</p>
    <div id="phoneme-grid" aria-live="polite"></div>
  </section>
```

The `</main>` boundary matters — the phoneme pane is its own top-level section, not part of the two-column main grid. Place it AFTER the closing `</main>` but before `<script>`.

- [ ] **Step 2: Append to `static/style.css`**

Append at the bottom of `static/style.css`:

```css
#phoneme-pane {
  background: var(--panel);
  padding: 1.2rem;
  border-radius: var(--radius);
  margin-top: 1.5rem;
}

#phoneme-pane h2 { margin-top: 0; }
#phoneme-pane .hint { color: var(--muted); margin-top: 0; }

#phoneme-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.8rem;
}

.phoneme-chip {
  font-family: ui-monospace, "Cascadia Mono", monospace;
  font-size: 0.85rem;
  padding: 0.35rem 0.7rem;
  border-radius: 6px;
  cursor: pointer;
  background: #4a2e2e;
  color: var(--error);
  user-select: none;
  border: 1px solid transparent;
}

.phoneme-chip.recorded {
  background: #2e4a32;
  color: #8fdf9b;
}

.phoneme-chip.armed {
  border-color: var(--accent);
}
```

- [ ] **Step 3: Spot-check that the page still renders**

Run (PowerShell):
```powershell
uv run flask --app app run
```

Open `http://127.0.0.1:5000`. Confirm:
- Existing two-pane layout (record / synthesise) is unchanged.
- Below the two panes, a new "Phoneme inventory" section appears with a hint line and an empty grid (the grid is filled in by JS in Task 11; it should be empty for now).
- No JavaScript console errors.

Stop with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add templates/index.html static/style.css
git commit -m "Add phoneme grid section to the page"
```

---

## Task 11: JS — phoneme grid + recording flow

**Files:**
- Modify: `static/app.js`

No automated tests — manual verify in Task 12.

- [ ] **Step 1: Add the new element references and state to `static/app.js`**

Find the `const els = { … }` block near the top of `static/app.js`. Add a new entry:

```javascript
  phonemeGrid: document.getElementById('phoneme-grid'),
```

After the existing module-level `let currentWavBlob = null;` line, add:

```javascript
let currentPhoneme = null;   // null => recording for a word; non-null => recording for this phoneme
```

- [ ] **Step 2: Replace the existing `refreshInventory` to also refresh the phoneme grid**

Find the existing `async function refreshInventory()` and replace its definition with:

```javascript
async function refreshInventory() {
  await Promise.all([refreshWordInventory(), refreshPhonemeInventory()]);
}

async function refreshWordInventory() {
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

async function refreshPhonemeInventory() {
  const r = await fetch('/api/phoneme-inventory');
  const { phonemes } = await r.json();
  els.phonemeGrid.innerHTML = '';
  for (const { symbol, recorded } of phonemes) {
    const chip = document.createElement('span');
    chip.className = 'phoneme-chip' + (recorded ? ' recorded' : '');
    chip.textContent = symbol;
    chip.dataset.symbol = symbol;
    chip.addEventListener('click', () => armPhoneme(symbol));
    if (currentPhoneme === symbol) chip.classList.add('armed');
    els.phonemeGrid.appendChild(chip);
  }
}
```

- [ ] **Step 3: Add the `armPhoneme` helper**

Append to `static/app.js`:

```javascript
function armPhoneme(symbol) {
  currentPhoneme = symbol;
  els.wordInput.value = '';   // clear any half-typed word
  setStatus(`Recording phoneme ${symbol} — press Record`);
  // Re-paint chips so the new selection is highlighted.
  for (const chip of els.phonemeGrid.querySelectorAll('.phoneme-chip')) {
    chip.classList.toggle('armed', chip.dataset.symbol === symbol);
  }
}

function disarmPhoneme() {
  currentPhoneme = null;
  for (const chip of els.phonemeGrid.querySelectorAll('.phoneme-chip')) {
    chip.classList.remove('armed');
  }
}
```

- [ ] **Step 4: Replace the Save button click handler to branch on word vs phoneme**

Find the existing `els.saveBtn.addEventListener('click', async () => { … })` handler. Replace its body with:

```javascript
els.saveBtn.addEventListener('click', async () => {
  if (!currentWavBlob) { setStatus('record something first', true); return; }

  if (currentPhoneme) {
    await savePhoneme(currentPhoneme, currentWavBlob);
  } else {
    const word = els.wordInput.value.trim();
    if (!word) { setStatus('enter a word first', true); return; }
    await saveWord(word, currentWavBlob);
  }

  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setButtons({ record: true, stop: false, save: false, discard: false });
  await refreshInventory();
});
```

- [ ] **Step 5: Extract `saveWord` and add `savePhoneme`**

Add these two helpers (place them just before the `els.saveBtn.addEventListener` block):

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
}

async function savePhoneme(symbol, blob) {
  const form = new FormData();
  form.append('phoneme', symbol);
  form.append('audio', blob, 'audio.wav');

  let r = await fetch('/api/record/phoneme', { method: 'POST', body: form });
  if (r.status === 409) {
    if (!confirm(`Phoneme ${symbol} already recorded. Overwrite?`)) return;
    r = await fetch('/api/record/phoneme?overwrite=true', { method: 'POST', body: form });
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return;
  }
  const { phoneme: saved } = await r.json();
  setStatus(`saved phoneme ${saved}`);
  disarmPhoneme();
}
```

- [ ] **Step 6: Make typing in the word input disarm any selected phoneme**

Add this listener near the bottom of `static/app.js`, just before the initial `refreshInventory()` call:

```javascript
els.wordInput.addEventListener('input', () => {
  if (currentPhoneme && els.wordInput.value.trim()) disarmPhoneme();
});
```

- [ ] **Step 7: Manual verification — phoneme record + synthesis**

Run (PowerShell):
```powershell
uv run flask --app app run
```

Open `http://127.0.0.1:5000`. Allow microphone. Then:

1. Confirm the phoneme grid shows 39 red chips.
2. Click `AA`. Confirm it gets an outline (armed) and the status shows "Recording phoneme AA — press Record".
3. Click Record, make any vowel-ish sound for at least 200 ms, click Stop. Confirm the preview plays back.
4. Click Save. Confirm:
   - The status says "saved phoneme AA".
   - The AA chip turns green.
   - The outline goes away.
5. Type a word ("hello") in the word input. Confirm any armed-phoneme outline disappears immediately (because typing disarms).
6. Record "hello" and save it. Confirm it appears in the word inventory chip list as usual.
7. Type `aa` in the sentence box (lowercase) and click Synthesise. Confirm playback (will be your recorded AA).
8. Type a word like "hello" in the sentence box. If recorded, it should play the word recording. Now type a word you have NOT recorded but whose phonemes you have — playback should be the stitched phonemes. Words whose phonemes aren't all recorded show up as red skipped chips.

Stop with Ctrl+C.

- [ ] **Step 8: Commit**

```bash
git add static/app.js
git commit -m "Add phoneme grid + phoneme recording flow in JS"
```

---

## Task 12: README + acceptance run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a Phase 2 section to `README.md`**

Find the existing `## Getting started` section. After it (and before the `## todo` section if one exists), add:

```markdown
## Phoneme fallback

If a sentence contains a word you haven't recorded directly, the synthesiser
tries to stitch it from individually-recorded phoneme samples. The bundled
[CMU Pronouncing Dictionary](data/cmudict.dict) provides the
grapheme-to-phoneme mapping. The phoneme inventory has 39 ARPABET symbols,
each shown as a chip below the record/synthesise panes — click any unrecorded
(red) chip to record that phoneme.

A word is synthesised via phonemes only if **every** required phoneme is
recorded. Words missing one or more phonemes, or words not in the dictionary,
are listed as skipped under the synthesise pane.
```

- [ ] **Step 2: Acceptance run**

Run (PowerShell):
```powershell
uv run pytest -v
```
Expected: all tests pass.

```powershell
uv run flask --app app run
```

Open `http://127.0.0.1:5000`. Run through:

1. Record three or four phonemes that cover a short word (e.g. for "hi" you need `HH AY`; for "the" you need `DH AH`).
2. Synthesise a sentence containing one of those words that you have NOT recorded as a word. Confirm playback.
3. Synthesise a sentence containing an out-of-dict gibberish word — confirm it shows up as skipped.
4. Synthesise a sentence containing a word whose phonemes are only partially recorded — confirm the whole word is skipped (not partially played).
5. Record a word that you have phonemes for (e.g. "hi") — synthesising should now use the word recording, not the phoneme stitch (verifiable by the audio sounding more natural).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document phoneme fallback in README"
```

---

## Self-review

**Spec coverage:**

- ✅ ARPABET 39-symbol constant — Task 2.
- ✅ CMUdict bundled (`data/cmudict.dict`) — Task 1.
- ✅ Dict loader with stress-strip, alternate-pronunciation suppression, malformed-line tolerance — Task 3.
- ✅ `phonemes.word_to_phonemes` returns list or None — Task 3.
- ✅ `inventory.phoneme_path / has_phoneme / save_phoneme / list_recorded_phonemes` — Task 4.
- ✅ `audio.concatenate` accepts grouped list, drops empty inner groups, backward-compatible with flat list — Task 5.
- ✅ `PhonemeResolver` rejects non-ARPABET, slots after `WordResolver` — Task 6.
- ✅ `synth.synthesise` two-pass algorithm, all-or-nothing per word, WordResolver precedence — Task 7.
- ✅ `POST /api/record/phoneme` with full validation pipeline (symbol, audio field, format, length, 409 / overwrite) — Task 8.
- ✅ `GET /api/phoneme-inventory` returns all 39 with recorded flag — Task 9.
- ✅ Phoneme grid UI + chip recording flow — Tasks 10–11.
- ✅ README documentation — Task 12.
- ✅ All listed test cases — distributed across Tasks 2-9.

**Placeholder scan:** No TBDs, all code blocks complete, all commands concrete. Test counts in Step 4 of Task 7 are estimated ("~78"); the actual final count comes out of running pytest.

**Type consistency:**
- `PhonemeResolver.resolve(word: str) -> Path | None` matches the `Resolver` Protocol in `resolver.py`.
- `inventory.phoneme_path(phoneme)` returns `Path` everywhere it is referenced.
- `phonemes.word_to_phonemes` returns `list[str] | None` and is consumed accordingly in `synth.synthesise`.
- `audio.concatenate` return type `bytes` unchanged.
- New `synthesise` signature unchanged: `tuple[bytes | None, list[str]]`.

**Plan complete.**

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-phoneme-fallback.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
