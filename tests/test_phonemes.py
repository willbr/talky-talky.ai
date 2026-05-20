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
