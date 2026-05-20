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
