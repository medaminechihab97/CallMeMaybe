"""Test integer constraints and final parameter validation."""

from src.call_once import validate_parameters
from src.number_parser import NumberParser
from src.parameter_parser import build_parameter_parser
from src.select_function import FunctionDefinition


def main() -> None:
    """Check integer syntax, fragment boundaries, and number regression."""
    # Standalone integer parsing.
    for text in ("0", "-0", "12", "-12", "123456789"):
        parser = NumberParser(integer_only=True).accept(text)
        assert parser.is_complete(), text
        assert parser.integer_only

    for text in ("1.5", "1.0", "1e3", "1E3", "+1", "01", "-01"):
        assert not NumberParser(integer_only=True).can_accept(text), text

    negative = NumberParser(integer_only=True).accept("-")
    assert not negative.is_complete()
    assert negative.accept("12").is_complete()

    # Integer mode must survive multiple fragments.
    parser = NumberParser(integer_only=True).accept("1")
    parser = parser.accept("2")
    assert parser.integer_only
    assert not parser.can_accept(".5")
    assert not parser.can_accept("e3")

    function = FunctionDefinition(
        name="fn_repeat",
        description="Repeat text a specified number of times.",
        parameters={
            "count": {"type": "integer"},
            "text": {"type": "string"},
        },
        returns={"type": "string"},
    )

    valid = [
        '{"count":0,"text":""}',
        '{"count":12,"text":"hello"}',
        '{ "count": -12, "text": "hello" }',
    ]

    for document in valid:
        result = validate_parameters(document, function)
        assert type(result["count"]) is int

        # Check every two-fragment split of the object.
        for split in range(len(document) + 1):
            object_parser = build_parameter_parser(function.parameters)

            for fragment in (document[:split], document[split:]):
                if fragment:
                    object_parser = object_parser.accept(fragment)

            assert object_parser.is_complete(), (document, split)

    for value in ('"12"', "true", "12.5", "12.0", "1e3", "01", "1 2"):
        document = '{"count":' + value + ',"text":"hello"}'

        object_parser = build_parameter_parser(function.parameters)
        assert not object_parser.can_accept(document), document

        try:
            validate_parameters(document, function)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid integer: {document}")

    # Regular number parameters still support decimals and exponents.
    for value in ("12.5", "-0.25", "1e3"):
        object_parser = build_parameter_parser({
            "amount": {"type": "number"},
        })
        assert object_parser.accept(
            '{"amount":' + value + "}"
        ).is_complete()

    print("Integer support tests passed.")


if __name__ == "__main__":
    main()