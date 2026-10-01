"""Test JSON strings and fragmented input without loading an LLM."""

import json

from src.string_parser import StringParser


def main() -> None:
    """Check valid strings, invalid input, and split escapes."""
    complete: list[str] = [
        '""',
        '"hello"',
        '"café 😀"',
        r'"say \"hi\""',
        r'"C:\\temp"',
        r'"\/\b\f\n\r\t"',
        r'"\u0041"',
        r'"\u00e9"',
        r'"\uD83D\uDE00"',
    ]

    incomplete: list[str] = [
        '"',
        '"hello',
        '"' + "\\",
        r'"\u',
        r'"\u0',
        r'"\u00',
        r'"\u004',
        r'"say \"hi\"',
    ]

    invalid: list[str] = [
        "hello",
        "'hello'",
        r'"\q"',
        r'"\x41"',
        r'"\u12G4"',
        r'"\u123"',
        '"line\nbreak"',
        '"tab\tinside"',
        '"nul\x00inside"',
        '"hello"x',
        '"hello" ',
    ]

    for text in complete:
        parser = StringParser().accept(text)
        assert parser.is_complete(), repr(text)
        assert isinstance(json.loads(text), str)

        # Check every two-fragment split, including inside escapes.
        for split in range(len(text) + 1):
            parser = StringParser()
            for fragment in (text[:split], text[split:]):
                if fragment:
                    parser = parser.accept(fragment)
            assert parser.is_complete(), (text, split)

    for text in incomplete:
        assert not StringParser().accept(text).is_complete(), repr(text)

    for text in invalid:
        assert not StringParser().can_accept(text), repr(text)

    assert not StringParser().is_complete()
    assert not StringParser().can_accept("")

    # A Unicode escape arriving across several fragments.
    parser = StringParser()
    for fragment in ['"', "\\", "u", "00", "41", '"']:
        parser = parser.accept(fragment)
        print(f"{fragment!r:>6} -> {parser.state}")
    assert parser.is_complete()

    # A failed trial must not modify the original parser.
    original = StringParser().accept('"hello')
    assert not original.can_accept(r'\q')
    assert original.state == "body"
    assert original.accept('"').is_complete()

    # No characters are accepted after the closing quote.
    finished = StringParser().accept('""')
    assert not finished.can_accept("x")

    print("String parser tests passed.")


if __name__ == "__main__":
    main()
