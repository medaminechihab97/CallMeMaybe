"""Test JSON whitespace and the rejected space-plus-quote token."""

import json
from typing import Any

from src.call_once import choose_token
from src.parameter_parser import build_parameter_parser


def main() -> None:
    """Check whitespace without weakening value or schema rules."""
    definitions: dict[str, dict[str, Any]] = {
        "name": {"type": "string"},
    }

    # Reproduce the parser state immediately before trace step 4.
    parser = build_parameter_parser(definitions)
    parser = parser.accept('{"name":')

    # Invented IDs, using the relevant fragments from the trace.
    fragments = {
        0: ' "',
        1: '"}',
    }

    # The highest-scoring legal continuation should no longer
    # be rejected because of the leading space.
    token_id, text, parser = choose_token(
        [36.663, 10.0],
        fragments,
        parser,
    )

    assert token_id == 0
    assert text == ' "'
    assert parser.accept('john"}').is_complete()

    valid = [
        '{"name": "john"}',
        ' \n{\n  "name" \t: "John Smith"\n}\r\n',
        '{"name":"  John Smith  "}',
        r'{"name": "John\nSmith"}',
    ]

    for document in valid:
        # Every two-fragment split must still work.
        for split in range(len(document) + 1):
            parser = build_parameter_parser(definitions)

            for fragment in (document[:split], document[split:]):
                if fragment:
                    parser = parser.accept(fragment)

            assert parser.is_complete(), (document, split)

    preserved = '{"name":"  John Smith  "}'
    assert json.loads(preserved)["name"] == "  John Smith  "

    # Do not permit whitespace to alter a required key.
    assert not build_parameter_parser(definitions).can_accept(
        '{"na me":"john"}'
    )

    # Literal control characters inside strings remain invalid.
    assert not build_parameter_parser(definitions).can_accept(
        '{"name":"John\nSmith"}'
    )
    assert not build_parameter_parser(definitions).can_accept(
        '{"name":"John\tSmith"}'
    )

    numeric = {"a": {"type": "number"}}
    assert build_parameter_parser(numeric).accept(
        '{"a": 12 \n}'
    ).is_complete()

    for document in (
        '{"a":1 2}',
        '{"a":1 .5}',
        '{"a":1. }',
        '{"a":- 1}',
    ):
        assert not build_parameter_parser(numeric).can_accept(document)

    boolean = {"enabled": {"type": "boolean"}}
    assert build_parameter_parser(boolean).accept(
        '{"enabled": true \n}'
    ).is_complete()
    assert not build_parameter_parser(boolean).can_accept(
        '{"enabled":tr ue}'
    )

    assert build_parameter_parser({}).accept(" \n{ \t} ").is_complete()

    print("Whitespace regression tests passed.")


if __name__ == "__main__":
    main()