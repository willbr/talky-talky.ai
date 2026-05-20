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
