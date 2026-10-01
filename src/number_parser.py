"""Recognize valid prefixes of a JSON number."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


NumberState = Literal[
    "start",
    "sign",
    "zero",
    "integer",
    "dot",
    "fraction",
    "exponent",
    "exponent_sign",
    "exponent_digits",
]

# For each state: character category -> next state.
TRANSITIONS: dict[NumberState, dict[str, NumberState]] = {
    "start": {
        "-": "sign",
        "0": "zero",
        "digit": "integer",
    },
    "sign": {
        "0": "zero",
        "digit": "integer",
    },
    "zero": {
        ".": "dot",
        "e": "exponent",
    },
    "integer": {
        "0": "integer",
        "digit": "integer",
        ".": "dot",
        "e": "exponent",
    },
    "dot": {
        "0": "fraction",
        "digit": "fraction",
    },
    "fraction": {
        "0": "fraction",
        "digit": "fraction",
        "e": "exponent",
    },
    "exponent": {
        "+": "exponent_sign",
        "-": "exponent_sign",
        "0": "exponent_digits",
        "digit": "exponent_digits",
    },
    "exponent_sign": {
        "0": "exponent_digits",
        "digit": "exponent_digits",
    },
    "exponent_digits": {
        "0": "exponent_digits",
        "digit": "exponent_digits",
    },
}

COMPLETE_STATES: set[NumberState] = {
    "zero",
    "integer",
    "fraction",
    "exponent_digits",
}


def character_category(character: str) -> str:
    """Group ASCII digits and exponent markers for the state table."""
    if "1" <= character <= "9":
        return "digit"
    if character in {"e", "E"}:
        return "e"
    return character


class NumberParser(BaseModel):
    """Track a JSON number prefix without converting it to a float."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    state: NumberState = "start"
    integer_only: bool = False

    def is_complete(self) -> bool:
        """Return whether the number may legally end here."""
        return self.state in COMPLETE_STATES

    def accept(self, text: str) -> "NumberParser":
        """Consume a nonempty fragment or reject it entirely."""
        if not text:
            raise ValueError("An empty fragment makes no progress.")

        state = self.state

        for character in text:
            category = character_category(character)
            if self.integer_only and category in {".", "e"}:
                raise ValueError(
                    "Integer values cannot contain decimals or exponents."
                )
            next_state = TRANSITIONS[state].get(category)

            if next_state is None:
                raise ValueError(
                    f"Character {character!r} is invalid "
                    f"in number state {state!r}."
                )

            state = next_state

        return NumberParser(
            state=state,
            integer_only=self.integer_only,
        )

    def can_accept(self, text: str) -> bool:
        """Check a fragment without changing the current parser."""
        try:
            self.accept(text)
        except ValueError:
            return False
        return True
