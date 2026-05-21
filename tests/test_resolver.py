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


def test_phoneme_resolver_hit(temp_inventory):
    vid = inventory.save_variant("AA", b"fakewav", source="isolated", prev=None, next=None)
    r = resolver.PhonemeResolver()
    assert r.resolve("AA") == temp_inventory / "phonemes" / "aa" / f"{vid}.wav"


def test_phoneme_resolver_case_insensitive(temp_inventory):
    vid = inventory.save_variant("AA", b"fakewav", source="isolated", prev=None, next=None)
    expected = temp_inventory / "phonemes" / "aa" / f"{vid}.wav"
    r = resolver.PhonemeResolver()
    assert r.resolve("aa") == expected
    assert r.resolve("Aa") == expected


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
