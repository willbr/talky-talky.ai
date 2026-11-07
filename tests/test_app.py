import base64
import io
import wave

import numpy as np
import pytest

import app as app_module
import audio
import inventory
import phonemes
import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _make_wav_bytes(n_samples=audio.SAMPLE_RATE, amplitude=10000, sample_rate=audio.SAMPLE_RATE):
    samples = np.ones(n_samples, dtype=np.int16) * amplitude
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _seed_samples(kind, label, samples, **kw):
    kw.setdefault("source", "isolated")
    return store.add_unit(kind, label, audio.encode_wav(samples), **kw)


def _seed_word(name, duration_seconds=1.0, amplitude=10000, **kw):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    return _seed_samples("word", name, np.ones(n, dtype=np.int16) * amplitude, **kw)


def _seed_phoneme(symbol, duration_seconds=0.4, amplitude=10000, **kw):
    n = int(audio.SAMPLE_RATE * duration_seconds)
    return _seed_samples("phoneme", symbol, np.ones(n, dtype=np.int16) * amplitude, **kw)


def _post_record(client, word, wav):
    return client.post(
        "/api/record",
        data={"word": word, "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )


def _post_record_phoneme(client, symbol, wav):
    return client.post(
        "/api/record/phoneme",
        data={"phoneme": symbol, "audio": (io.BytesIO(wav), "audio.wav")},
        content_type="multipart/form-data",
    )


def _ms(path):
    samples, _ = audio.read_wav(path)
    return int(len(samples) * 1000 / audio.SAMPLE_RATE)


# --- index & inventory ---------------------------------------------------------

def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"talky-talky.ai" in r.data


def test_inventory_empty(client):
    r = client.get("/api/inventory")
    assert r.status_code == 200
    assert r.get_json() == {"words": []}


def test_inventory_lists_saved_words(client):
    _seed_word("hello")
    _seed_word("hello")
    _seed_word("apple")
    r = client.get("/api/inventory")
    assert r.get_json() == {"words": ["apple", "hello"]}


# --- recording ------------------------------------------------------------------

def test_record_saves_valid_wav(client):
    r = _post_record(client, "Hello!", _make_wav_bytes())
    assert r.status_code == 200
    body = r.get_json()
    assert body["word"] == "hello"
    units = store.list_units("word", "hello")
    assert [u["id"] for u in units] == [body["unit_id"]]
    assert units[0]["source"] == "isolated"


def test_record_existing_word_appends_copy(client):
    first = _post_record(client, "hello", _make_wav_bytes())
    second = _post_record(client, "hello", _make_wav_bytes())
    assert first.status_code == 200 and second.status_code == 200
    assert first.get_json()["unit_id"] != second.get_json()["unit_id"]
    assert len(store.list_units("word", "hello")) == 2


def test_record_ignores_overwrite_flag(client):
    _post_record(client, "hello", _make_wav_bytes())
    r = client.post(
        "/api/record?overwrite=true",
        data={"word": "hello", "audio": (io.BytesIO(_make_wav_bytes()), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert len(store.list_units("word", "hello")) == 2


def test_record_rejects_wrong_sample_rate(client):
    r = _post_record(client, "hello", _make_wav_bytes(sample_rate=44100))
    assert r.status_code == 400


def test_record_rejects_invalid_wav(client):
    r = _post_record(client, "hello", b"not a wav")
    assert r.status_code == 400


def test_record_rejects_empty_normalised_word(client):
    r = _post_record(client, "...", _make_wav_bytes())
    assert r.status_code == 400


def test_record_rejects_too_short(client):
    r = _post_record(client, "hi", _make_wav_bytes(n_samples=100))
    assert r.status_code == 400


def test_record_phoneme_saves_valid_wav(client):
    r = _post_record_phoneme(client, "AA", _make_wav_bytes())
    assert r.status_code == 200
    body = r.get_json()
    assert body["phoneme"] == "AA"
    assert [u["id"] for u in store.list_units("phoneme", "AA")] == [body["unit_id"]]


def test_record_phoneme_case_insensitive(client):
    r = _post_record_phoneme(client, "aa", _make_wav_bytes())
    assert r.status_code == 200
    assert r.get_json()["phoneme"] == "AA"
    assert len(store.list_units("phoneme", "AA")) == 1


def test_record_phoneme_rejects_unknown_symbol(client):
    r = _post_record_phoneme(client, "ZZ", _make_wav_bytes())
    assert r.status_code == 400
    assert set(r.get_json()["allowed"]) == phonemes.PHONEMES_SET


def test_record_phoneme_rejects_empty_field(client):
    assert _post_record_phoneme(client, "", _make_wav_bytes()).status_code == 400


def test_record_phoneme_rejects_wrong_sample_rate(client):
    assert _post_record_phoneme(client, "AA", _make_wav_bytes(sample_rate=44100)).status_code == 400


def test_record_phoneme_rejects_too_short(client):
    assert _post_record_phoneme(client, "AA", _make_wav_bytes(n_samples=100)).status_code == 400


def test_record_phoneme_appends_new_copy(client):
    a = _post_record_phoneme(client, "AA", _make_wav_bytes()).get_json()["unit_id"]
    b = _post_record_phoneme(client, "AA", _make_wav_bytes()).get_json()["unit_id"]
    assert a != b
    assert [u["id"] for u in store.list_units("phoneme", "AA")] == [a, b]


def test_phoneme_inventory_lists_all_39(client):
    data = client.get("/api/phoneme-inventory").get_json()
    assert [e["symbol"] for e in data["phonemes"]] == list(phonemes.PHONEMES)
    assert all(e["recorded"] is False for e in data["phonemes"])


def test_phoneme_inventory_reflects_saved_phonemes(client):
    _seed_phoneme("AA")
    _seed_phoneme("B")
    by_symbol = {e["symbol"]: e["recorded"]
                 for e in client.get("/api/phoneme-inventory").get_json()["phonemes"]}
    assert by_symbol["AA"] is True
    assert by_symbol["B"] is True
    assert by_symbol["AE"] is False


def test_phoneme_inventory_includes_example_words(client):
    examples = {e["symbol"]: e["example"]
                for e in client.get("/api/phoneme-inventory").get_json()["phonemes"]}
    assert len(examples) == 39
    assert all(isinstance(v, str) and v for v in examples.values())


# --- units routes ---------------------------------------------------------------

def test_units_list_and_audio(client):
    a = _seed_word("hello")
    b = _seed_word("hello", source="sentence", prev="say", next="world", position="mid")
    r = client.get("/api/units?kind=word&label=Hello!")
    assert r.status_code == 200
    body = r.get_json()
    assert body["kind"] == "word" and body["label"] == "hello"
    assert [u["id"] for u in body["units"]] == [a, b]
    assert body["units"][1]["position"] == "mid"

    r = client.get(f"/api/units/{a}/audio")
    assert r.status_code == 200
    assert r.mimetype == "audio/wav"
    assert r.data == store.unit_path(a).read_bytes()


def test_units_list_phoneme_label_is_uppercased(client):
    uid = _seed_phoneme("AA")
    body = client.get("/api/units?kind=phoneme&label=aa").get_json()
    assert body["label"] == "AA"
    assert [u["id"] for u in body["units"]] == [uid]


def test_units_list_400s(client):
    assert client.get("/api/units?kind=bogus&label=x").status_code == 400
    assert client.get("/api/units?kind=word&label=...").status_code == 400
    assert client.get("/api/units?kind=word").status_code == 400
    assert client.get("/api/units?kind=phoneme&label=QQ").status_code == 400


def test_unit_delete_204_then_404(client):
    uid = _seed_word("hello")
    assert client.delete(f"/api/units/{uid}").status_code == 204
    assert not store.unit_path(uid).exists()
    assert client.delete(f"/api/units/{uid}").status_code == 404


def test_unit_routes_reject_bad_ids(client, tmp_path):
    keep = _seed_word("hello")
    (tmp_path / "secret.wav").write_bytes(b"x")
    assert client.delete("/api/units/..%2F..%2Fsecret").status_code == 404
    assert client.delete("/api/units/nope").status_code == 404
    assert client.get("/api/units/nope/audio").status_code == 404
    assert client.get("/api/units/..%2Fsecret/audio").status_code == 404
    assert (tmp_path / "secret.wav").exists()
    assert store.unit_path(keep).exists()


def test_unit_audio_404_when_file_missing(client):
    uid = _seed_word("hello")
    store.unit_path(uid).unlink()
    assert client.get(f"/api/units/{uid}/audio").status_code == 404


def test_old_variant_routes_are_gone(client):
    assert client.get("/api/phoneme/variants?symbol=AA").status_code in (404, 405)


# --- synthesis --------------------------------------------------------------------

def test_synthesize_returns_audio_and_skipped(client):
    _seed_word("hello")
    _seed_word("world")
    data = client.post("/api/synthesize", json={"sentence": "Hello, world!"}).get_json()
    assert data["skipped"] == []
    assert base64.b64decode(data["audio"]).startswith(b"RIFF")


def test_synthesize_reports_skipped(client):
    _seed_word("hello")
    data = client.post("/api/synthesize", json={"sentence": "hello mars"}).get_json()
    assert data["skipped"] == ["mars"]
    assert data["audio"] is not None


def test_synthesize_all_missing(client):
    data = client.post("/api/synthesize", json={"sentence": "mars venus"}).get_json()
    assert data["audio"] is None
    assert data["skipped"] == ["mars", "venus"]


def test_synthesize_returns_units_and_spectrogram(client):
    _seed_word("hello")
    _seed_word("world")
    data = client.post("/api/synthesize", json={"sentence": "hello world"}).get_json()
    assert [u["token"] for u in data["units"]] == ["hello", "world"]
    assert data["units"][0]["start_ms"] == 0
    assert data["units"][0]["end_ms"] == data["units"][1]["start_ms"]
    spec = data["spectrogram"]
    assert spec["n_frames"] > 0
    assert spec["n_freqs"] == 257
    assert len(base64.b64decode(spec["data"])) == spec["n_frames"] * spec["n_freqs"]


def test_synthesize_units_carry_unit_id(client):
    hello = _seed_word("hello")
    world = _seed_word("world")
    data = client.post("/api/synthesize", json={"sentence": "hello world"}).get_json()
    assert [u["unit_id"] for u in data["units"]] == [hello, world]


def test_synthesize_no_spectrogram_when_no_audio(client):
    data = client.post("/api/synthesize", json={"sentence": "mars venus"}).get_json()
    assert data["audio"] is None
    assert data["spectrogram"] is None
    assert data["units"] == []


def test_synthesize_empty_sentence_400(client):
    assert client.post("/api/synthesize", json={"sentence": ""}).status_code == 400


def test_synthesize_missing_field_400(client):
    assert client.post("/api/synthesize", json={}).status_code == 400


# --- extraction (word editor) ---------------------------------------------------

def test_extract_propose_returns_audio_and_boundaries(client):
    uid = _seed_word("hello")
    r = client.post("/api/extract/propose", json={"word": "hello"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["word"] == "hello"
    assert data["unit_id"] == uid
    assert data["audio_duration_ms"] > 0
    assert data["audio"]
    assert len(data["phonemes"]) >= 1
    for entry in data["phonemes"]:
        assert {"symbol", "example", "recorded", "prior_ms"} <= set(entry)
        assert isinstance(entry["recorded"], bool)
        assert isinstance(entry["prior_ms"], int) and entry["prior_ms"] > 0
    assert len(data["spans"]) == len(data["phonemes"])
    for prev, curr in zip(data["spans"], data["spans"][1:]):
        assert prev["end_ms"] <= curr["start_ms"]


def test_extract_propose_defaults_to_most_recent_copy(client):
    _seed_word("hello")
    newer = _seed_word("hello")
    assert client.post("/api/extract/propose", json={"word": "hello"}).get_json()["unit_id"] == newer


def test_extract_propose_404_when_word_not_recorded(client):
    assert client.post("/api/extract/propose", json={"word": "ghost"}).status_code == 404


def test_extract_propose_404_for_unknown_unit_id(client):
    _seed_word("hello")
    r = client.post("/api/extract/propose", json={"unit_id": "20200101T000000000000"})
    assert r.status_code == 404


def test_extract_propose_404_for_phoneme_unit_id(client):
    uid = _seed_phoneme("AA")
    assert client.post("/api/extract/propose", json={"unit_id": uid}).status_code == 404


def test_extract_propose_400_when_word_oov(client):
    _seed_word("xyzqqqzz")
    r = client.post("/api/extract/propose", json={"word": "xyzqqqzz"})
    assert r.status_code == 400
    assert "dictionary" in r.get_json()["error"].lower()


def test_extract_propose_400_when_word_missing_field(client):
    assert client.post("/api/extract/propose", json={}).status_code == 400


def test_extract_propose_normalises_word(client):
    _seed_word("hello")
    r = client.post("/api/extract/propose", json={"word": "  HELLO!  "})
    assert r.status_code == 200
    assert r.get_json()["word"] == "hello"


def _propose(client, **body):
    r = client.post("/api/extract/propose", json=body)
    assert r.status_code == 200
    return r.get_json()


def test_extract_save_writes_selected_phonemes(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    r = client.post("/api/extract/save", json={
        "word": "hello", "spans": prop["spans"], "save": [True] * len(prop["phonemes"]),
    })
    assert r.status_code == 200
    body = r.get_json()
    assert {e["symbol"] for e in body["saved"]} == {e["symbol"] for e in prop["phonemes"]}
    assert body["skipped"] == []
    for entry in body["saved"]:
        assert store.get_unit(entry["unit_id"])["label"] == entry["symbol"]


def test_extract_save_records_source_unit(client):
    uid = _seed_word("hello")
    prop = _propose(client, unit_id=uid)
    phs = [e["symbol"] for e in prop["phonemes"]]
    r = client.post("/api/extract/save", json={
        "unit_id": uid, "spans": prop["spans"], "save": [True] * len(phs),
    })
    assert r.status_code == 200
    saved = r.get_json()["saved"]
    assert len(saved) == len(phs)
    for i, entry in enumerate(saved):
        unit = store.get_unit(entry["unit_id"])
        assert unit["source"] == "word"
        assert unit["source_unit_id"] == uid
        assert unit["source_word"] == "hello"
        assert unit["prev"] == (phs[i - 1] if i > 0 else None)
        assert unit["next"] == (phs[i + 1] if i < len(phs) - 1 else None)


def test_extract_save_skips_phonemes_with_save_false(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    flags = [True] * len(prop["phonemes"])
    flags[1] = False
    body = client.post("/api/extract/save", json={
        "word": "hello", "spans": prop["spans"], "save": flags,
    }).get_json()
    skipped_symbol = prop["phonemes"][1]["symbol"]
    assert skipped_symbol not in {e["symbol"] for e in body["saved"]}
    assert store.list_units("phoneme", skipped_symbol) == []


def test_extract_save_skips_short_slices(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    n = len(prop["phonemes"])
    spans = [dict(s) for s in prop["spans"]]
    spans[1]["end_ms"] = spans[1]["start_ms"] + 1
    for i in range(2, n):
        spans[i]["start_ms"] = max(spans[i]["start_ms"], spans[i - 1]["end_ms"])
        spans[i]["end_ms"] = max(spans[i]["end_ms"], spans[i]["start_ms"] + 1)
    body = client.post("/api/extract/save", json={
        "word": "hello", "spans": spans, "save": [True] * n,
    }).get_json()
    assert prop["phonemes"][1]["symbol"] in [s["symbol"] for s in body["skipped"]]
    assert "too short" in [s["reason"] for s in body["skipped"]]


def test_extract_save_rejects_overlapping_spans(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    spans = [dict(s) for s in prop["spans"]]
    spans[1]["start_ms"] = spans[0]["start_ms"]
    r = client.post("/api/extract/save", json={
        "word": "hello", "spans": spans, "save": [True] * len(spans),
    })
    assert r.status_code == 400


def test_extract_save_rejects_length_mismatch(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    r = client.post("/api/extract/save", json={
        "word": "hello", "spans": prop["spans"], "save": [True],
    })
    assert r.status_code == 400


def test_extract_save_appends_new_copy(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    first_symbol = prop["phonemes"][0]["symbol"]
    existing = _seed_phoneme(first_symbol)
    flags = [False] * len(prop["phonemes"])
    flags[0] = True
    body = client.post("/api/extract/save", json={
        "word": "hello", "spans": prop["spans"], "save": flags,
    }).get_json()
    new = body["saved"][0]["unit_id"]
    assert new != existing
    assert {u["id"] for u in store.list_units("phoneme", first_symbol)} == {existing, new}


def test_extract_save_404_when_word_not_recorded(client):
    r = client.post("/api/extract/save", json={
        "word": "ghost",
        "spans": [{"start_ms": 0, "end_ms": 100}, {"start_ms": 100, "end_ms": 200}],
        "save": [True, True],
    })
    assert r.status_code == 404


def test_extract_save_normalises_word(client):
    _seed_word("hello")
    prop = _propose(client, word="hello")
    r = client.post("/api/extract/save", json={
        "word": "  HELLO!  ", "spans": prop["spans"], "save": [True] * len(prop["phonemes"]),
    })
    assert r.status_code == 200


def test_extract_targets_named_copy(client):
    older = _seed_word("hello", duration_seconds=1.0)
    newer = _seed_word("hello", duration_seconds=1.0)
    assert _propose(client, unit_id=older)["unit_id"] == older
    r = client.post("/api/extract/trim", json={"unit_id": older, "start_ms": 100, "end_ms": 400})
    assert r.status_code == 200
    body = r.get_json()
    assert body["unit_id"] == older and body["word"] == "hello"
    assert 295 <= _ms(store.unit_path(older)) <= 305
    assert 995 <= _ms(store.unit_path(newer)) <= 1005
    assert store.get_unit(older)["duration_ms"] == body["duration_ms"]


def test_delete_last_word_copy_keeps_cut_phonemes(client):
    uid = _seed_word("hello")
    prop = _propose(client, unit_id=uid)
    saved = client.post("/api/extract/save", json={
        "unit_id": uid, "spans": prop["spans"], "save": [True] * len(prop["phonemes"]),
    }).get_json()["saved"]
    assert client.delete(f"/api/units/{uid}").status_code == 204
    assert client.get("/api/inventory").get_json() == {"words": []}
    for entry in saved:
        unit = store.get_unit(entry["unit_id"])
        assert unit is not None
        assert unit["source_word"] == "hello"


# --- word trim -----------------------------------------------------------------

def test_trim_crops_recording(client):
    samples = np.full(audio.SAMPLE_RATE, 100, dtype=np.int16)
    samples[int(0.45 * audio.SAMPLE_RATE):int(0.55 * audio.SAMPLE_RATE)] = 20000
    uid = _seed_samples("word", "hello", samples)
    r = client.post("/api/extract/trim", json={"word": "hello", "start_ms": 300, "end_ms": 700})
    assert r.status_code == 200
    body = r.get_json()
    assert body["word"] == "hello"
    assert body["unit_id"] == uid
    assert 395 <= body["duration_ms"] <= 405
    cropped, _ = audio.read_wav(store.unit_path(uid))
    assert 0.395 * audio.SAMPLE_RATE <= len(cropped) <= 0.405 * audio.SAMPLE_RATE
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1


def test_trim_404_when_word_not_recorded(client):
    r = client.post("/api/extract/trim", json={"word": "ghost", "start_ms": 0, "end_ms": 200})
    assert r.status_code == 404


def test_trim_400_when_range_inverted(client):
    _seed_word("hello")
    r = client.post("/api/extract/trim", json={"word": "hello", "start_ms": 500, "end_ms": 200})
    assert r.status_code == 400
    assert "greater" in r.get_json()["error"].lower()


def test_trim_400_when_too_short(client):
    _seed_word("hello")
    r = client.post("/api/extract/trim", json={"word": "hello", "start_ms": 100, "end_ms": 150})
    assert r.status_code == 400
    assert "too short" in r.get_json()["error"].lower()


def test_trim_400_when_out_of_range(client):
    _seed_word("hello", duration_seconds=1.0)
    r = client.post("/api/extract/trim", json={"word": "hello", "start_ms": 0, "end_ms": 5000})
    assert r.status_code == 400
    assert "out of range" in r.get_json()["error"].lower()


def test_trim_renormalises_loudness(client):
    uid = _seed_samples("word", "hello", np.full(audio.SAMPLE_RATE, 1000, dtype=np.int16))
    r = client.post("/api/extract/trim", json={"word": "hello", "start_ms": 200, "end_ms": 800})
    assert r.status_code == 200
    cropped, _ = audio.read_wav(store.unit_path(uid))
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1


# --- phoneme editor -------------------------------------------------------------

def test_phoneme_propose_returns_audio(client):
    uid = _seed_phoneme("AA")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AA"
    assert data["unit_id"] == uid
    assert data["audio_duration_ms"] > 0
    assert base64.b64decode(data["audio"]).startswith(b"RIFF")


def test_phoneme_propose_404_when_not_recorded(client):
    assert client.post("/api/phoneme/propose", json={"symbol": "AA"}).status_code == 404


def test_phoneme_propose_400_for_unknown_symbol(client):
    assert client.post("/api/phoneme/propose", json={"symbol": "QQ"}).status_code == 400


def test_phoneme_propose_accepts_unit_id(client):
    older = _seed_phoneme("AA")
    _seed_phoneme("AA", source="word", source_word="hello", prev="HH", next="L")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA", "unit_id": older})
    assert r.status_code == 200
    assert r.get_json()["unit_id"] == older


def test_phoneme_propose_404_for_unknown_unit_id(client):
    _seed_phoneme("AA")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA", "unit_id": "nope"})
    assert r.status_code == 404


def test_phoneme_propose_404_for_other_symbols_unit(client):
    _seed_phoneme("AA")
    other = _seed_phoneme("B")
    r = client.post("/api/phoneme/propose", json={"symbol": "AA", "unit_id": other})
    assert r.status_code == 404


def test_phoneme_trim_crops_recording(client):
    samples = np.full(audio.SAMPLE_RATE, 100, dtype=np.int16)
    samples[int(0.3 * audio.SAMPLE_RATE):int(0.5 * audio.SAMPLE_RATE)] = 20000
    uid = _seed_samples("phoneme", "AA", samples)
    r = client.post("/api/phoneme/trim", json={"symbol": "AA", "start_ms": 200, "end_ms": 600})
    assert r.status_code == 200
    body = r.get_json()
    assert body["symbol"] == "AA"
    assert body["unit_id"] == uid
    assert 395 <= body["duration_ms"] <= 405
    cropped, _ = audio.read_wav(store.unit_path(uid))
    assert int(np.max(np.abs(cropped))) >= audio.PEAK_TARGET - 1
    assert store.get_unit(uid)["duration_ms"] == body["duration_ms"]


def test_phoneme_trim_targets_named_unit(client):
    target = _seed_phoneme("AA", duration_seconds=1.0)
    other = _seed_phoneme("AA", duration_seconds=1.0)
    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "unit_id": target, "start_ms": 100, "end_ms": 400,
    })
    assert r.status_code == 200
    assert r.get_json()["unit_id"] == target
    assert 295 <= _ms(store.unit_path(target)) <= 305
    assert 995 <= _ms(store.unit_path(other)) <= 1005


def test_phoneme_trim_rejects_word_unit_id(client):
    _seed_phoneme("AA")
    word = _seed_word("hello")
    r = client.post("/api/phoneme/trim", json={
        "symbol": "AA", "unit_id": word, "start_ms": 100, "end_ms": 400,
    })
    assert r.status_code == 404
    assert 995 <= _ms(store.unit_path(word)) <= 1005


def test_phoneme_trim_404_when_not_recorded(client):
    r = client.post("/api/phoneme/trim", json={"symbol": "AA", "start_ms": 0, "end_ms": 100})
    assert r.status_code == 404


def test_phoneme_trim_400_when_too_short(client):
    _seed_phoneme("AA")
    r = client.post("/api/phoneme/trim", json={"symbol": "AA", "start_ms": 0, "end_ms": 10})
    assert r.status_code == 400
    assert "too short" in r.get_json()["error"].lower()


def test_phoneme_trim_400_when_inverted(client):
    _seed_phoneme("AA")
    r = client.post("/api/phoneme/trim", json={"symbol": "AA", "start_ms": 200, "end_ms": 100})
    assert r.status_code == 400


# --- reference sentences --------------------------------------------------------

def test_sentence_audio_returns_match(client):
    sid = store.save_sentence("Hello, world!", _make_wav_bytes())
    r = client.post("/api/sentence/audio", json={"sentence": "hello world"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["found"] is True
    assert data["id"] == sid
    assert base64.b64decode(data["audio"]).startswith(b"RIFF")
    assert data["spectrogram"]["n_frames"] > 0
    assert data["spectrogram"]["n_freqs"] == 257


def test_sentence_audio_returns_not_found(client):
    r = client.post("/api/sentence/audio", json={"sentence": "nobody recorded this"})
    assert r.status_code == 200
    assert r.get_json() == {"found": False}


def test_sentence_audio_400_when_sentence_missing(client):
    assert client.post("/api/sentence/audio", json={}).status_code == 400


def test_sentence_record_saves_pair(client):
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "hello world", "audio": (io.BytesIO(_make_wav_bytes()), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    sid = r.get_json()["id"]
    assert store.list_sentences() == [{"id": sid, "text": "hello world"}]
    assert store.find_sentence_by_text("hello world").stem == sid


def test_sentence_record_rejects_empty_sentence(client):
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "  ", "audio": (io.BytesIO(_make_wav_bytes()), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_sentence_record_rejects_wrong_sample_rate(client):
    r = client.post(
        "/api/sentence/record",
        data={"sentence": "hi", "audio": (io.BytesIO(_make_wav_bytes(sample_rate=44100)), "audio.wav")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_sentences_list(client):
    client.post(
        "/api/sentence/record",
        data={"sentence": "hello", "audio": (io.BytesIO(_make_wav_bytes()), "audio.wav")},
        content_type="multipart/form-data",
    )
    data = client.get("/api/sentences").get_json()
    assert len(data["sentences"]) == 1
    assert data["sentences"][0]["text"] == "hello"
