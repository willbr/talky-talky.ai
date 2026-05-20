import synth


def test_tokenise_splits_and_lowercases():
    assert synth.tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    assert synth.tokenise("Hello, world!") == ["hello", "world"]
    assert synth.tokenise("Don't stop.") == ["dont", "stop"]


def test_tokenise_collapses_whitespace():
    assert synth.tokenise("  hello   world  ") == ["hello", "world"]


def test_tokenise_empty_returns_empty_list():
    assert synth.tokenise("") == []
    assert synth.tokenise("   ") == []
    assert synth.tokenise("...!?") == []
