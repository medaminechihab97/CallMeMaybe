"""Test schema-constrained parameter objects without an LLM."""

import json
from typing import Any

from src.parameter_parser import build_parameter_parser


def main() -> None:
    """Check boundaries, types, schema constraints, and fragments."""
    definitions: dict[str, dict[str, Any]] = {
        "count": {"type": "number"},
        "name": {"type": "string"},
        "enabled": {"type": "boolean"},
    }

    valid: list[str] = [
        '{"count":12,"name":"Ali","enabled":true}',
        '{"count":-1.25e+2,"name":"","enabled":false}',
        r'{"count":0,"name":"say \"hi\"","enabled":true}',
        r'{"count":2,"name":"\u0041","enabled":false}',
    ]

    for text in valid:
        parser = build_parameter_parser(definitions).accept(text)
        assert parser.is_complete(), text

        result = json.loads(text)
        assert set(result) == set(definitions)
        assert type(result["count"]) in (int, float)
        assert type(result["name"]) is str
        assert type(result["enabled"]) is bool

        # Every two-fragment split must produce the same result.
        for split in range(len(text) + 1):
            parser = build_parameter_parser(definitions)
            for fragment in (text[:split], text[split:]):
                if fragment:
                    parser = parser.accept(fragment)
            assert parser.is_complete(), (text, split)

    invalid: list[str] = [
        '{}',
        '{"count":"12","name":"Ali","enabled":true}',
        '{"count":true,"name":"Ali","enabled":true}',
        '{"count":01,"name":"Ali","enabled":true}',
        '{"count":12.,"name":"Ali","enabled":true}',
        '{"count":12,"name":3,"enabled":true}',
        '{"count":12,"name":"Ali","enabled":"true"}',
        '{"count":12,"name":"Ali","enabled":True}',
        '{"count":12,"name":"Ali","enabled":true,}',
        '{"count":12,"name":"Ali","enabled":true,"extra":1}',
        '{"count":12,"name":"Ali"}',
        '{"count":12,"count":13,"enabled":true}',
        '{"name":"Ali","count":12,"enabled":true}',
        '{"count": 12,"name":"Ali","enabled":true}',
    ]

    for text in invalid:
        parser = build_parameter_parser(definitions)
        assert not parser.can_accept(text), text

    # Cross a numeric boundary, a key, and a string in one fragment.
    parser = build_parameter_parser(definitions)
    parser = parser.accept('{"count":12')
    assert not parser.is_complete()

    parser = parser.accept(',"name":"Ali","enabled":tr')
    assert not parser.is_complete()

    parser = parser.accept("ue}")
    assert parser.is_complete()
    assert not parser.can_accept("x")

    # Failed trial parsing does not modify the original state.
    original = build_parameter_parser(definitions)
    assert not original.can_accept('{"count":01')
    assert original.part_index == 0

    # No parameters means exactly an empty object.
    empty = build_parameter_parser({})
    assert empty.accept("{}").is_complete()

    # Property names are serialized, not interpolated unsafely.
    special: dict[str, dict[str, Any]] = {
        'say"hi': {"type": "string"},
    }
    text = json.dumps({'say"hi': "ok"}, separators=(",", ":"))
    assert build_parameter_parser(special).accept(text).is_complete()

    # Unsupported types and constraints must fail explicitly.
    unsupported: list[dict[str, dict[str, Any]]] = [
        {"items": {"type": "array"}},
        {"count": {"type": "number", "minimum": 0}},
    ]

    for definition in unsupported:
        try:
            build_parameter_parser(definition)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported schema was accepted.")

    print("Parameter-object parser tests passed.")


if __name__ == "__main__":
    main()