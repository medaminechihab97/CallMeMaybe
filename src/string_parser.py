"""Recognize valid prefixes of a JSON string."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


StringState = Literal[
    "start",
    "body",
    "escape",
    "unicode_1",
    "unicode_2",
    "unicode_3",
    "unicode_4",
    "done",
]

SIMPLE_ESCAPES: frozenset[str] = frozenset('"\\/bfnrt')
HEX_DIGITS: frozenset[str] = frozenset(
    "0123456789abcdefABCDEF"
)

UNICODE_NEXT: dict[StringState, StringState] = {
    "unicode_1": "unicode_2",
    "unicode_2": "unicode_3",
    "unicode_3": "unicode_4",
    "unicode_4": "body",
}


def advance_string(
    state: StringState,
    character: str,
) -> StringState:
    """Consume one character or reject an invalid transition."""
    if len(character) != 1:
        raise ValueError("Expected exactly one character.")

    if state == "start":
        if character == '"':
            return "body"

    elif state == "body":
        if character == '"':
            return "done"
        if character == "\\":
            return "escape"
        if ord(character) >= 0x20:
            return "body"

    elif state == "escape":
        if character in SIMPLE_ESCAPES:
            return "body"
        if character == "u":
            return "unicode_1"

    elif state in UNICODE_NEXT:
        if character in HEX_DIGITS:
            return UNICODE_NEXT[state]

    raise ValueError(
        f"Character {character!r} is invalid "
        f"in string state {state!r}."
    )


class StringParser(BaseModel):
    """Track JSON string syntax without decoding escape sequences."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    state: StringState = "start"

    def is_complete(self) -> bool:
        """Return whether the closing quote has been consumed."""
        return self.state == "done"

    def accept(self, text: str) -> "StringParser":
        """Consume a nonempty fragment or reject it entirely."""
        if not text:
            raise ValueError("An empty fragment makes no progress.")

        state = self.state

        for character in text:
            state = advance_string(state, character)

        return StringParser(state=state)

    def can_accept(self, text: str) -> bool:
        """Check a fragment without changing the current parser."""
        try:
            self.accept(text)
        except ValueError:
            return False
        return True
