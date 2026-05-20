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
