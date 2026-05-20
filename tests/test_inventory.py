import pytest

import inventory


def test_normalise_lowercases():
    assert inventory.normalise("Hello") == "hello"


def test_normalise_strips_punctuation():
    assert inventory.normalise("Hello!") == "hello"
    assert inventory.normalise("Don't") == "dont"
    assert inventory.normalise("(world)") == "world"


def test_normalise_strips_surrounding_whitespace():
    assert inventory.normalise("  hello  ") == "hello"


def test_normalise_empty_input_returns_empty():
    assert inventory.normalise("") == ""
    assert inventory.normalise("...") == ""
    assert inventory.normalise("!?") == ""


@pytest.fixture
def temp_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path)
    return tmp_path


def test_path_uses_normalised_filename(temp_inventory):
    assert inventory.path("Hello!") == temp_inventory / "hello.wav"


def test_save_writes_bytes_to_disk(temp_inventory):
    wav_bytes = b"RIFF....fakewav....data...."
    inventory.save("hello", wav_bytes)
    assert (temp_inventory / "hello.wav").read_bytes() == wav_bytes


def test_has_reflects_existence(temp_inventory):
    assert not inventory.has("hello")
    inventory.save("hello", b"x")
    assert inventory.has("hello")


def test_list_words_returns_sorted_unique(temp_inventory):
    inventory.save("world", b"x")
    inventory.save("hello", b"x")
    inventory.save("apple", b"x")
    assert inventory.list_words() == ["apple", "hello", "world"]


def test_save_creates_inventory_dir_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "newdir"
    monkeypatch.setattr(inventory, "INVENTORY_DIR", target)
    inventory.save("hello", b"x")
    assert target.is_dir()
    assert (target / "hello.wav").exists()


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
