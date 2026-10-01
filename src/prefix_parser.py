"""Track valid continuations of a finite set of token sequences."""

from pydantic import BaseModel, ConfigDict, Field


class PrefixParser(BaseModel):
    """Keep the unconsumed suffixes of all matching candidates."""

    model_config = ConfigDict(extra="forbid", strict=True)

    remaining: list[list[int]] = Field(min_length=1)

    def allowed_token_ids(self) -> set[int]:
        """Return tokens that can continue a matching candidate."""
        return {
            sequence[0]
            for sequence in self.remaining
            if sequence
        }

    def is_complete(self) -> bool:
        """Return whether a complete candidate has been consumed."""
        return any(not sequence for sequence in self.remaining)

    def accept(self, token_id: int) -> "PrefixParser":
        """Consume a valid token and return the updated parser."""
        matching_suffixes = [
            sequence[1:]
            for sequence in self.remaining
            if sequence and sequence[0] == token_id
        ]

        if not matching_suffixes:
            raise ValueError(
                f"Token {token_id} is not a valid continuation."
            )

        return PrefixParser(remaining=matching_suffixes)
