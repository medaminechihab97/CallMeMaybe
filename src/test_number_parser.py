"""Test JSON number prefixes without loading an LLM."""

from src.number_parser import NumberParser


def main() -> None:
    """Check complete numbers, incomplete prefixes, and failures."""
    complete = [
        "0", "-0", "42", "-12.5", "0.25",
        "1e3", "1E-3", "-12.5e+3",
    ]
    incomplete = [
        "-", "12.", "1e", "1e+", "1e-",
    ]
    invalid = [
        "01", "-01", "+1", ".5", "1..2",
        "1e+-2", "NaN", "Infinity", "１２",
    ]

    for text in complete:
        assert NumberParser().accept(text).is_complete(), text

    for text in incomplete:
        assert not NumberParser().accept(text).is_complete(), text

    for text in invalid:
        assert not NumberParser().can_accept(text), text

    assert not NumberParser().is_complete()
    assert not NumberParser().can_accept("")

    # Simulate fragments arriving across several generation steps.
    parser = NumberParser()

    for fragment in ["-", "12", ".5", "e+", "3"]:
        parser = parser.accept(fragment)
        print(f"{fragment!r:>5} -> {parser.state}")

    assert parser.is_complete()

    # Reject the entire fragment, leaving the original state intact.
    original = NumberParser().accept("12")
    assert not original.can_accept(".5x")
    assert original.state == "integer"

    # A complete number can still be extended.
    assert original.is_complete()
    assert original.can_accept("3")
    assert original.can_accept(".5")

    print("Number parser tests passed.")


if __name__ == "__main__":
    main()