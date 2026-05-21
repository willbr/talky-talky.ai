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
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    (tmp_path / "phonemes" / "aa.wav").write_bytes(b"x")
    (tmp_path / "phonemes" / "b.wav").write_bytes(b"x")

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
        "boundaries_ms": prop["boundaries_ms"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert set(body["saved"]) == set(entry["symbol"] for entry in prop["phonemes"])
    assert body["skipped"] == []
    for entry in prop["phonemes"]:
        assert (tmp_path / "phonemes" / f"{entry['symbol'].lower()}.wav").exists()


def test_extract_save_skips_phonemes_with_save_false(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n_phonemes = len(prop["phonemes"])
    save_flags = [True] * n_phonemes
    save_flags[1] = False

    r = client.post("/api/extract/save", json={
        "word": "hello",
        "boundaries_ms": prop["boundaries_ms"],
        "save": save_flags,
    })
    assert r.status_code == 200
    body = r.get_json()
    skipped_symbol = prop["phonemes"][1]["symbol"]
    assert skipped_symbol not in body["saved"]
    assert not (tmp_path / "phonemes" / f"{skipped_symbol.lower()}.wav").exists()


def test_extract_save_skips_short_slices(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    n = len(prop["phonemes"])
    bounds = list(prop["boundaries_ms"])
    if n >= 2:
        bounds[2] = bounds[1] + 1
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
        bounds[1], bounds[2] = bounds[2], bounds[1]

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
        "save": [True],
    })
    assert r.status_code == 400


def test_extract_save_overwrites_existing_phoneme(client, tmp_path):
    _seed_word(tmp_path, "hello")
    pr = client.post("/api/extract/propose", json={"word": "hello"})
    prop = pr.get_json()
    first_symbol = prop["phonemes"][0]["symbol"]
    (tmp_path / "phonemes").mkdir(exist_ok=True)
    target = tmp_path / "phonemes" / f"{first_symbol.lower()}.wav"
    target.write_bytes(b"OLDOLDOLDOLD")

    save_flags = [False] * len(prop["phonemes"])
    save_flags[0] = True

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
