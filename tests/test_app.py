import io
import wave

import pytest
import numpy as np

import app as app_module
import audio
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
