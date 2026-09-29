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
    kind: Literal["number", "string", "boolean"]


ValueParser: TypeAlias = NumberParser | StringParser | BooleanParser
Part: TypeAlias = str | ParameterSpec


def make_value_parser(spec: ParameterSpec) -> ValueParser:
    """Create a fresh parser for a parameter's value."""
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
        """Consume one character, including value-boundary handling."""
        if self.is_complete():
            raise ValueError("Unexpected text after the object.")

        part = self.parts[self.part_index]

        if isinstance(part, str):
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
            active = make_value_parser(part)

        # Numbers do not have an explicit closing character.
        # If this character cannot extend the number, it may
        # belong to the next object part instead.
        if isinstance(active, NumberParser):
            if not active.can_accept(character):
                if not active.is_complete():
                    raise ValueError(
                        f"Incomplete number for {part.name!r}."
                    )

                # Reprocess the SAME character as object syntax.
                return self._next_part()._consume(character)

        updated = active.accept(character)

        # Strings close with a quote; booleans finish when their
        # full literal has been consumed.
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
        parts.append(encoded_name + ":")
        parts.append(spec)

    parts.append("}")

    return ParameterObjectParser(parts=tuple(parts))