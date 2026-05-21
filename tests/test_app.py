import base64
import io
import wave

import pytest
import numpy as np

import app as app_module
import audio
import extraction  # noqa: F401  (ensures module is importable in test env)
import inventory
import phonemes
import synth


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


def test_synthesize_returns_units_and_spectrogram(client, tmp_path):
    for word in ("hello", "world"):
        samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
        audio.write_wav(tmp_path / f"{word}.wav", samples)

    r = client.post("/api/synthesize", json={"sentence": "hello world"})
    assert r.status_code == 200
    data = r.get_json()

    assert [u["token"] for u in data["units"]] == ["hello", "world"]
    assert data["units"][0]["start_ms"] == 0
    assert data["units"][0]["end_ms"] == data["units"][1]["start_ms"]

    spec = data["spectrogram"]
    assert spec is not None
    assert spec["n_frames"] > 0
    assert spec["n_freqs"] == 257  # 512 // 2 + 1
    raw = base64.b64decode(spec["data"])
    assert len(raw) == spec["n_frames"] * spec["n_freqs"]


def test_synthesize_no_spectrogram_when_no_audio(client):
    r = client.post("/api/synthesize", json={"sentence": "mars venus"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["audio"] is None
    assert data["spectrogram"] is None
    assert data["units"] == []


def test_synthesize_empty_sentence_400(client):
    r = client.post("/api/synthesize", json={"sentence": ""})
    assert r.status_code == 400


def test_synthesize_missing_field_400(client):
    r = client.post("/api/synthesize", json={})
    assert r.status_code == 400


def test_record_phoneme_saves_valid_wav(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["phoneme"] == "AA"
    assert body["variant_id"]
    assert (tmp_path / "phonemes" / "aa" / f"{body['variant_id']}.wav").exists()


def test_record_phoneme_case_insensitive(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/record/phoneme",
        data={"phoneme": "aa", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["phoneme"] == "AA"
    assert (tmp_path / "phonemes" / "aa" / f"{body['variant_id']}.wav").exists()


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
    assert set(body["allowed"]) == phonemes.PHONEMES_SET


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


def test_record_phoneme_appends_new_variant(client, tmp_path):
    """Each /api/record/phoneme adds a new variant — never clobbers the
    prior take. The user manages duplicates by deleting from the chip."""
    wav = _make_wav_bytes()
    first = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    second = client.post(
        "/api/record/phoneme",
        data={"phoneme": "AA", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert first.status_code == 200 and second.status_code == 200
    vid1 = first.get_json()["variant_id"]
    vid2 = second.get_json()["variant_id"]
    assert vid1 != vid2
    assert (tmp_path / "phonemes" / "aa" / f"{vid1}.wav").exists()
    assert (tmp_path / "phonemes" / "aa" / f"{vid2}.wav").exists()


def test_phoneme_inventory_lists_all_39(client):
    r = client.get("/api/phoneme-inventory")
    assert r.status_code == 200
    data = r.get_json()
    assert "phonemes" in data
    assert len(data["phonemes"]) == 39
    symbols = [entry["symbol"] for entry in data["phonemes"]]
    assert symbols == list(phonemes.PHONEMES)  # canonical order
    assert all(entry["recorded"] is False for entry in data["phonemes"])


def test_phoneme_inventory_reflects_saved_phonemes(client, tmp_path):
    wav = _make_wav_bytes()
    inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)
    inventory.save_variant("B", wav, source="isolated", prev=None, next=None)

    r = client.get("/api/phoneme-inventory")
    assert r.status_code == 200
    by_symbol = {e["symbol"]: e["recorded"] for e in r.get_json()["phonemes"]}
    assert by_symbol["AA"] is True
    assert by_symbol["B"] is True
    assert by_symbol["AE"] is False


def test_phoneme_inventory_includes_example_words(client):
    r = client.get("/api/phoneme-inventory")
    data = r.get_json()
    examples = {e["symbol"]: e["example"] for e in data["phonemes"]}
    # Every phoneme has a non-empty example word
    assert len(examples) == 39
    assert all(isinstance(v, str) and v for v in examples.values())
    # Spot-check a couple of canonical mappings
    assert examples["AA"] == "AA" or examples["AA"]  # non-empty
    assert examples["ZH"]


def test_extract_propose_returns_audio_and_boundaries(client, tmp_path):
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "hello.wav", samples)

    r = client.post("/api/extract/propose", json={"word": "hello"})
    assert r.status_code == 200
    data = r.get_json()

    assert data["word"] == "hello"
    assert data["audio_duration_ms"] > 0
    assert data["audio"]  # non-empty base64
    assert len(data["phonemes"]) >= 1
    for entry in data["phonemes"]:
        assert "symbol" in entry
        assert "example" in entry
        assert "recorded" in entry
        assert isinstance(entry["recorded"], bool)
        assert "prior_ms" in entry
        assert isinstance(entry["prior_ms"], int) and entry["prior_ms"] > 0
    assert len(data["spans"]) == len(data["phonemes"])
    for s in data["spans"]:
        assert s["start_ms"] <= s["end_ms"]
    for prev, curr in zip(data["spans"], data["spans"][1:]):
        assert prev["end_ms"] <= curr["start_ms"]


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


def _seed_word(tmp_path, name, duration_seconds=1.0, amplitude=10000):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    samples = np.ones(n, dtype=np.int16) * amplitude
    audio.write_wav(tmp_path / f"{name}.wav", samples)


def test_extract_save_writes_selected_phonemes(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n_phonemes = len(prop["phonemes"])
    save_flags = [True] * n_phonemes

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": prop["spans"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    saved_symbols = {entry["symbol"] for entry in body["saved"]}
    assert saved_symbols == set(entry["symbol"] for entry in prop["phonemes"])
    assert all(entry.get("variant_id") for entry in body["saved"])
    assert body["skipped"] == []
    for entry in body["saved"]:
        assert inventory.variant_path(entry["symbol"], entry["variant_id"]).exists()


def test_extract_save_skips_phonemes_with_save_false(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n_phonemes = len(prop["phonemes"])
    save_flags = [True] * n_phonemes
    save_flags[1] = False

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": prop["spans"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    skipped_symbol = prop["phonemes"][1]["symbol"]
    saved_symbols = {entry["symbol"] for entry in body["saved"]}
    assert skipped_symbol not in saved_symbols
    assert not (tmp_path / "phonemes" / skipped_symbol.lower()).exists()


def test_extract_save_skips_short_slices(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n = len(prop["phonemes"])
    spans = [dict(s) for s in prop["spans"]]
    if n >= 2:
        spans[1]["end_ms"] = spans[1]["start_ms"] + 1
        for i in range(2, n):
            if spans[i]["start_ms"] < spans[i - 1]["end_ms"]:
                spans[i]["start_ms"] = spans[i - 1]["end_ms"]
            if spans[i]["end_ms"] < spans[i]["start_ms"]:
                spans[i]["end_ms"] = spans[i]["start_ms"]
    save_flags = [True] * n

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": spans,
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    skipped_symbols = [s["symbol"] for s in body["skipped"]]
    skipped_reasons = [s["reason"] for s in body["skipped"]]
    assert prop["phonemes"][1]["symbol"] in skipped_symbols
    assert "too short" in skipped_reasons


def test_extract_save_rejects_overlapping_spans(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    spans = [dict(s) for s in prop["spans"]]
    # Force overlap: pull span[1] back into span[0].
    spans[1]["start_ms"] = spans[0]["start_ms"]

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": spans,
        "save": [True] * len(spans),
    })
    assert r.status_code == 400


def test_extract_save_rejects_length_mismatch(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": prop["spans"],
        "save": [True],
    })
    assert r.status_code == 400


def test_extract_save_appends_new_variant(client, tmp_path):
    """Saving a phoneme from a word never overwrites a prior recording —
    each save appends a new variant. The earlier recording remains in
    the inventory unchanged."""
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    first_symbol = prop["phonemes"][0]["symbol"]
    # Pre-existing isolated take of the same phoneme.
    existing_id = inventory.save_variant(
        first_symbol, _make_wav_bytes(),
        source="isolated", prev=None, next=None,
    )

    save_flags = [False] * len(prop["phonemes"])
    save_flags[0] = True
    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": prop["spans"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    new_id = body["saved"][0]["variant_id"]
    assert new_id != existing_id
    # Both variants exist on disk; the pre-existing isolated take survives.
    assert inventory.variant_path(first_symbol, existing_id).exists()
    assert inventory.variant_path(first_symbol, new_id).exists()


def test_extract_save_404_when_word_not_recorded(client):
    r = client.post("/api/extract/save", json={
        "word": "ghost",
        "spans": [{"start_ms": 0, "end_ms": 100}, {"start_ms": 100, "end_ms": 200}],
        "save": [True, True],
    })
    assert r.status_code == 404


def test_extract_save_normalises_word(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    r = client.post("/api/extract/save", json={
        "word": "  HELLO!  ",
        "spans": prop["spans"],
        "save": [True] * len(prop["phonemes"]),
    })
    assert r.status_code == 200


def test_trim_crops_recording(client, tmp_path):
    # 1 s of low-amplitude noise with a louder burst around 500 ms.
    n = audio.SAMPLE_RATE
    samples = np.full(n, 100, dtype=np.int16)
    burst_start = int(0.45 * audio.SAMPLE_RATE)
    burst_end = int(0.55 * audio.SAMPLE_RATE)
    samples[burst_start:burst_end] = 20000
    audio.write_wav(tmp_path / "hello.wav", samples)

    r = client.post("/api/extract/trim", json={
        "word": "hello",
        "start_ms": 300,
        "end_ms": 700,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["word"] == "hello"
    assert 395 <= body["duration_ms"] <= 405

    cropped, _ = audio.read_wav(tmp_path / "hello.wav")
    assert 0.395 * audio.SAMPLE_RATE <= len(cropped) <= 0.405 * audio.SAMPLE_RATE
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1


def test_trim_404_when_word_not_recorded(client):
    r = client.post("/api/extract/trim", json={
        "word": "ghost",
        "start_ms": 0,
        "end_ms": 200,
    })
    assert r.status_code == 404


def test_trim_400_when_range_inverted(client, tmp_path):
    _seed_word(tmp_path, "hello")
    r = client.post("/api/extract/trim", json={
        "word": "hello",
        "start_ms": 500,
        "end_ms": 200,
    })
    assert r.status_code == 400
    assert "greater" in r.get_json()["error"].lower()


def test_trim_400_when_too_short(client, tmp_path):
    _seed_word(tmp_path, "hello")
    r = client.post("/api/extract/trim", json={
        "word": "hello",
        "start_ms": 100,
        "end_ms": 150,
    })
    assert r.status_code == 400
    assert "too short" in r.get_json()["error"].lower()


def test_trim_400_when_out_of_range(client, tmp_path):
    _seed_word(tmp_path, "hello", duration_seconds=1.0)
    r = client.post("/api/extract/trim", json={
        "word": "hello",
        "start_ms": 0,
        "end_ms": 5000,
    })
    assert r.status_code == 400
    assert "out of range" in r.get_json()["error"].lower()


def test_trim_renormalises_loudness(client, tmp_path):
    n = audio.SAMPLE_RATE
    samples = np.full(n, 1000, dtype=np.int16)  # quiet — well below PEAK_TARGET
    audio.write_wav(tmp_path / "hello.wav", samples)

    r = client.post("/api/extract/trim", json={
        "word": "hello",
        "start_ms": 200,
        "end_ms": 800,
    })
    assert r.status_code == 200

    cropped, _ = audio.read_wav(tmp_path / "hello.wav")
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1


def _seed_phoneme(tmp_path, symbol, duration_seconds=0.4, amplitude=10000):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    samples = np.ones(n, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return inventory.save_variant(
        symbol, buf.getvalue(),
        source="isolated", prev=None, next=None,
    )


def test_phoneme_propose_returns_audio(client, tmp_path):
    _seed_phoneme(tmp_path, "AA")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AA"
    assert data["audio_duration_ms"] > 0
    assert base64.b64decode(data["audio"]).startswith(b"RIFF")


def test_phoneme_propose_404_when_not_recorded(client):
    r = client.post("/api/phoneme/propose", json={"symbol": "AA"})
    assert r.status_code == 404


def test_phoneme_propose_400_for_unknown_symbol(client):
    r = client.post("/api/phoneme/propose", json={"symbol": "QQ"})
    assert r.status_code == 400


def test_phoneme_trim_crops_recording(client, tmp_path):
    n = audio.SAMPLE_RATE
    samples = np.full(n, 100, dtype=np.int16)
    burst_start = int(0.3 * audio.SAMPLE_RATE)
    burst_end = int(0.5 * audio.SAMPLE_RATE)
    samples[burst_start:burst_end] = 20000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    variant_id = inventory.save_variant(
        "AA", buf.getvalue(),
        source="isolated", prev=None, next=None,
    )

    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA",
        "start_ms": 200,
        "end_ms": 600,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["symbol"] == "AA"
    assert body["variant_id"] == variant_id
    assert 395 <= body["duration_ms"] <= 405

    cropped, _ = audio.read_wav(inventory.variant_path("AA", variant_id))
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1


def test_phoneme_trim_404_when_not_recorded(client):
    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "start_ms": 0, "end_ms": 100,
    })
    assert r.status_code == 404


def test_phoneme_trim_400_when_too_short(client, tmp_path):
    _seed_phoneme(tmp_path, "AA")
    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "start_ms": 0, "end_ms": 10,
    })
    assert r.status_code == 400
    assert "too short" in r.get_json()["error"].lower()


def test_sentence_audio_returns_match(client, tmp_path):
    # Seed a reference recording on disk in the format save_sentence uses.
    (tmp_path / "sentences").mkdir(exist_ok=True)
    samples = np.ones(audio.SAMPLE_RATE, dtype=np.int16) * 10000
    audio.write_wav(tmp_path / "sentences" / "ref1.wav", samples)
    (tmp_path / "sentences" / "ref1.txt").write_text("Hello, world!", encoding="utf-8")

    # Looked up by normalised text — punctuation/case differences match.
    r = client.post("/api/sentence/audio", json={"sentence": "hello world"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["found"] is True
    assert data["id"] == "ref1"
    assert base64.b64decode(data["audio"]).startswith(b"RIFF")
    assert data["spectrogram"]["n_frames"] > 0
    assert data["spectrogram"]["n_freqs"] == 257


def test_sentence_audio_returns_not_found(client, tmp_path):
    (tmp_path / "sentences").mkdir(exist_ok=True)
    r = client.post("/api/sentence/audio", json={"sentence": "nobody recorded this"})
    assert r.status_code == 200
    assert r.get_json() == {"found": False}


def test_sentence_audio_400_when_sentence_missing(client):
    r = client.post("/api/sentence/audio", json={})
    assert r.status_code == 400


def test_extract_save_records_prev_next_per_position(client, tmp_path):
    """For a multi-phoneme word, the first phoneme variant has prev=None,
    the last has next=None, and any middle phoneme has both neighbours
    set."""
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    phs = [entry["symbol"] for entry in prop["phonemes"]]
    save_flags = [True] * len(phs)
    r = client.post("/api/extract/save", json={
        "word": "hello",
        "spans": prop["spans"],
        "save": save_flags,
    })
    assert r.status_code == 200
    saved = r.get_json()["saved"]
    # Pick out variant metadata for each saved phoneme.
    for i, entry in enumerate(saved):
        meta_list = inventory.list_variants(entry["symbol"])
        meta = next(m for m in meta_list if m["id"] == entry["variant_id"])
        expected_prev = phs[i - 1] if i > 0 else None
        expected_next = phs[i + 1] if i < len(phs) - 1 else None
        assert meta["prev"] == expected_prev
        assert meta["next"] == expected_next
        assert meta["source"] == "hello"


def test_variants_route_lists_all(client, tmp_path):
    wav = _make_wav_bytes()
    vid1 = inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)
    vid2 = inventory.save_variant("AA", wav, source="hello", prev="HH", next="L")
    r = client.get("/api/phoneme/variants?symbol=AA")
    assert r.status_code == 200
    body = r.get_json()
    assert body["symbol"] == "AA"
    ids = [v["id"] for v in body["variants"]]
    assert set(ids) == {vid1, vid2}


def test_variants_route_400_unknown_symbol(client):
    r = client.get("/api/phoneme/variants?symbol=QQ")
    assert r.status_code == 400


def test_variants_route_400_missing_symbol(client):
    r = client.get("/api/phoneme/variants")
    assert r.status_code == 400


def test_variant_delete_204_then_404(client, tmp_path):
    vid = inventory.save_variant("AA", _make_wav_bytes(),
                                 source="isolated", prev=None, next=None)
    r = client.delete(f"/api/phoneme/variants?symbol=AA&id={vid}")
    assert r.status_code == 204
    assert not inventory.variant_path("AA", vid).exists()
    r = client.delete(f"/api/phoneme/variants?symbol=AA&id={vid}")
    assert r.status_code == 404


def test_phoneme_propose_accepts_variant_id(client, tmp_path):
    """Two variants exist; passing variant_id selects the older one
    instead of defaulting to most-recent."""
    wav = _make_wav_bytes()
    older = inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)
    inventory.save_variant("AA", wav, source="hello", prev="HH", next="L")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA", "variant_id": older})
    assert r.status_code == 200
    assert r.get_json()["variant_id"] == older


def test_phoneme_propose_404_for_unknown_variant_id(client, tmp_path):
    inventory.save_variant("AA", _make_wav_bytes(),
                           source="isolated", prev=None, next=None)
    r = client.post("/api/phoneme/propose", json={"symbol": "AA", "variant_id": "nope"})
    assert r.status_code == 404


def test_phoneme_trim_targets_named_variant(client, tmp_path):
    """Trimming with variant_id overwrites only that specific variant;
    the other variant is untouched."""
    n = audio.SAMPLE_RATE  # 1 s recordings
    samples = np.ones(n, dtype=np.int16) * 10000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.CHANNELS)
        w.setsampwidth(audio.SAMPLE_WIDTH)
        w.setframerate(audio.SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    wav = buf.getvalue()
    target = inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)
    other = inventory.save_variant("AA", wav, source="isolated", prev=None, next=None)

    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "variant_id": target,
        "start_ms": 100, "end_ms": 400,
    })
    assert r.status_code == 200
    assert r.get_json()["variant_id"] == target

    target_samples, _ = audio.read_wav(inventory.variant_path("AA", target))
    other_samples, _ = audio.read_wav(inventory.variant_path("AA", other))
    target_ms = int(len(target_samples) * 1000 / audio.SAMPLE_RATE)
    other_ms = int(len(other_samples) * 1000 / audio.SAMPLE_RATE)
    assert 295 <= target_ms <= 305
    assert 995 <= other_ms <= 1005


def test_phoneme_trim_400_when_inverted(client, tmp_path):
    _seed_phoneme(tmp_path, "AA")
    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "start_ms": 200, "end_ms": 100,
    })
    assert r.status_code == 400


def test_sentence_record_saves_pair(client, tmp_path):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "hello world", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    sid = r.get_json()["id"]
    assert (tmp_path / "sentences" / f"{sid}.wav").exists()
    assert (tmp_path / "sentences" / f"{sid}.txt").read_text(encoding="utf-8") == "hello world"


def test_sentence_record_rejects_empty_sentence(client):
    wav = _make_wav_bytes()
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "  ", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_sentence_record_rejects_wrong_sample_rate(client):
    wav = _make_wav_bytes(sample_rate=44100)
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "hi", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_sentences_list(client, tmp_path):
    wav = _make_wav_bytes()
    client.post(
        "/api/sentence/record",
        data={"sentence": "hello", "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )
    r = client.get("/api/sentences")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["sentences"]) == 1
    assert data["sentences"][0]["text"] == "hello"
