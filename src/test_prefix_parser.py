"""Run a small, model-free test of the prefix parser."""

from src.prefix_parser import PrefixParser


def main() -> None:
    """Check branching, completion, and invalid-token rejection."""
    parser = PrefixParser(remaining=[[10, 20, 30], [10, 21, 30]])

    assert parser.allowed_token_ids() == {10}
    assert not parser.is_complete()

    parser = parser.accept(10)
    assert parser.allowed_token_ids() == {20, 21}

    try:
        parser.accept(99)
    except ValueError:
        pass
    else:
        raise AssertionError("The parser accepted an invalid token.")

    parser = parser.accept(21)
    assert parser.allowed_token_ids() == {30}

    parser = parser.accept(30)
    assert parser.is_complete()
    assert parser.allowed_token_ids() == set()

    print("Parser tests passed.")


if __name__ == "__main__":
    main()
