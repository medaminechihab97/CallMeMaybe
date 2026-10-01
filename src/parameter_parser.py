"""Recognize schema-constrained, compact parameter objects."""

import json
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from src.boolean_parser import BooleanParser
from src.number_parser import NumberParser
from src.string_parser import StringParser


class ParameterSpec(BaseModel):
    """Describe one required parameter and its supported type."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    name: str
    kind: Literal["number", "string", "boolean", "integer"]


ValueParser: TypeAlias = NumberParser | StringParser | BooleanParser
Part: TypeAlias = str | ParameterSpec


def make_value_parser(spec: ParameterSpec) -> ValueParser:
    """Create a fresh parser for a parameter's value."""
    if spec.kind == "integer":
        return NumberParser(integer_only=True)
    if spec.kind == "number":
        return NumberParser()
    if spec.kind == "string":
        return StringParser()
    return BooleanParser()


class ParameterObjectParser(BaseModel):
    """Track an object using fixed literals and typed value parsers."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    parts: tuple[Part, ...]
    part_index: int = 0
    literal_offset: int = 0
    value: ValueParser | None = None

    def is_complete(self) -> bool:
        """Return whether the entire object has been consumed."""
        return self.part_index == len(self.parts)

    def _next_part(self) -> "ParameterObjectParser":
        """Advance to the next part and reset local progress."""
        return self.model_copy(
            update={
                "part_index": self.part_index + 1,
                "literal_offset": 0,
                "value": None,
            }
        )

    def _consume(self, character: str) -> "ParameterObjectParser":
        """Consume one character, allowing JSON boundary whitespace."""
        whitespace = character in " \t\r\n"

        if self.is_complete():
            if whitespace:
                return self
            raise ValueError("Unexpected text after the object.")

        part = self.parts[self.part_index]

        if isinstance(part, str):
            # Allow whitespace before a key or structural literal,
            # but never skip characters inside a quoted key.
            if self.literal_offset == 0 and whitespace:
                return self

            expected = part[self.literal_offset]

            if character != expected:
                raise ValueError(
                    f"Expected {expected!r}, got {character!r}."
                )

            offset = self.literal_offset + 1

            if offset == len(part):
                return self._next_part()

            return self.model_copy(
                update={"literal_offset": offset}
            )

        active = self.value

        if active is None:
            # Before a value begins, whitespace is formatting.
            if whitespace:
                return self
            active = make_value_parser(part)

        if isinstance(active, NumberParser):
            if not active.can_accept(character):
                if not active.is_complete():
                    raise ValueError(
                        f"Incomplete number for {part.name!r}."
                    )

                # End the number, then process the same character
                # as object syntax. Whitespace must terminate a
                # number: "1 2" must never become "12".
                return self._next_part()._consume(character)

        # Inside a string, spaces belong to the value; unescaped
        # control characters remain forbidden by StringParser.
        # Inside a boolean, whitespace cannot split its spelling.
        updated = active.accept(character)

        if not isinstance(updated, NumberParser):
            if updated.is_complete():
                return self._next_part()

        return self.model_copy(update={"value": updated})

    def accept(self, text: str) -> "ParameterObjectParser":
        """Consume a whole fragment or reject it without mutation."""
        if not text:
            raise ValueError("An empty fragment makes no progress.")

        parser = self

        for character in text:
            parser = parser._consume(character)

        return parser

    def can_accept(self, text: str) -> bool:
        """Check a fragment without changing this parser."""
        try:
            self.accept(text)
        except ValueError:
            return False
        return True


def build_parameter_parser(
    parameters: dict[str, dict[str, Any]],
) -> ParameterObjectParser:
    """Compile assignment-style parameter definitions into a parser."""
    parts: list[Part] = ["{"]

    for index, (name, metadata) in enumerate(parameters.items()):
        unsupported = set(metadata) - {"type", "description"}

        if unsupported:
            raise ValueError(
                f"Unsupported constraints for {name!r}: "
                f"{sorted(unsupported)}"
            )

        spec = ParameterSpec.model_validate(
            {
                "name": name,
                "kind": metadata.get("type"),
            }
        )

        if index:
            parts.append(",")

        encoded_name = json.dumps(spec.name, ensure_ascii=True)
        parts.append(encoded_name)
        parts.append(":")
        parts.append(spec)

    parts.append("}")

    return ParameterObjectParser(parts=tuple(parts))
