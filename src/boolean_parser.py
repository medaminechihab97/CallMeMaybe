"""Recognize valid prefixes of a JSON boolean."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


BooleanPrefix = Literal[
    "", "t", "tr", "tru", "true",
    "f", "fa", "fal", "fals", "false",
]


class BooleanParser(BaseModel):
    """Track progress through either true or false."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    prefix: BooleanPrefix = ""

    def is_complete(self) -> bool:
        """Return whether a complete boolean has been consumed."""
        return self.prefix in {"true", "false"}

    def accept(self, text: str) -> "BooleanParser":
        """Consume a nonempty valid continuation."""
        if not text:
            raise ValueError("An empty fragment makes no progress.")

        candidate = self.prefix + text

        if not any(
            literal.startswith(candidate)
            for literal in ("true", "false")
        ):
            raise ValueError(f"Invalid boolean prefix: {candidate!r}")

        return BooleanParser.model_validate({"prefix": candidate})

    def can_accept(self, text: str) -> bool:
        """Check a fragment without changing this parser."""
        try:
            self.accept(text)
        except ValueError:
            return False
        return True