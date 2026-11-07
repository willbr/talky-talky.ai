import pytest

import inventory
import resolver
import store
from resolver import Context


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def _word(label, **kw):
    kw.setdefault("source", "isolated")
    return store.add_unit("word", label, b"fakewav", **kw)


def _phoneme(symbol, **kw):
    kw.setdefault("source", "isolated")
    return store.add_unit("phoneme", symbol, b"fakewav", **kw)


def test_word_resolver_hit(temp_inventory):
    uid = _word("hello")
    assert resolver.WordResolver().resolve("hello", Context())["id"] == uid


def test_word_resolver_miss(temp_inventory):
    assert resolver.WordResolver().resolve("missing", Context()) is None


def test_word_resolver_normalises_input(temp_inventory):
    uid = _word("hello")
    assert resolver.WordResolver().resolve("Hello!", Context())["id"] == uid


def test_word_resolver_uses_context(temp_inventory):
    contextual = _word("hello", source="sentence", next="world", position="mid")
    isolated = _word("hello")
    r = resolver.WordResolver()
    assert r.resolve("hello", Context(next="world", position="mid"))["id"] == contextual
    assert r.resolve("hello", Context())["id"] == isolated


def test_resolver_chain_contains_word_resolver():
    assert any(isinstance(r, resolver.WordResolver) for r in resolver.RESOLVERS)


def test_phoneme_resolver_hit(temp_inventory):
    uid = _phoneme("AA")
    assert resolver.PhonemeResolver().resolve("AA", Context())["id"] == uid


def test_phoneme_resolver_case_insensitive(temp_inventory):
    uid = _phoneme("AA")
    r = resolver.PhonemeResolver()
    assert r.resolve("aa", Context())["id"] == uid
    assert r.resolve("Aa", Context())["id"] == uid


def test_phoneme_resolver_miss_no_file(temp_inventory):
    assert resolver.PhonemeResolver().resolve("AA", Context()) is None


def test_phoneme_resolver_rejects_non_arpabet(temp_inventory):
    _word("hello")
    assert resolver.PhonemeResolver().resolve("hello", Context()) is None


def test_resolvers_chain_order():
    assert len(resolver.RESOLVERS) == 2
    assert isinstance(resolver.RESOLVERS[0], resolver.WordResolver)
    assert isinstance(resolver.RESOLVERS[1], resolver.PhonemeResolver)
