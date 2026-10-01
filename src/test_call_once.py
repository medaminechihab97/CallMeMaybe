"""Test constrained token selection without loading model weights."""

from src.call_once import choose_token
from src.parameter_parser import build_parameter_parser


def main() -> None:
    """Reject a higher-scoring candidate with the wrong value type."""
    parser = build_parameter_parser({
        "a": {"type": "number"},
    })

    # Invented token IDs and fragments.
    fragments = {
        0: '{"a":',
        1: '"wrong"}',
        2: '12}',
    }

    token_id, _, parser = choose_token(
        [10.0, 2.0, 1.0],
        fragments,
        parser,
    )
    assert token_id == 0
    assert not parser.is_complete()

    # Token 1 scores highest, but a string is not allowed here.
    token_id, _, parser = choose_token(
        [0.0, 20.0, 5.0],
        fragments,
        parser,
    )
    assert token_id == 2
    assert parser.is_complete()

    print("Constrained selection test passed.")


if __name__ == "__main__":
    main()
