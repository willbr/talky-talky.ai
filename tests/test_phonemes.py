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
