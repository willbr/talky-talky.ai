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


def test_is_valid_id_accepts_timestamp_ids():
    assert inventory.is_valid_id("20261107T162337769846")
    assert inventory.is_valid_id("00000000T000000000000")


def test_is_valid_id_rejects_everything_else():
    for bad in ["", "nope", "../../hello", "..\\..\\hello",
                "20261107T162337769846/../x", "20261107T16233776984", None, 123]:
        assert not inventory.is_valid_id(bad)
