"""Validate batch inputs and publish complete JSON output safely."""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
)


class PromptRecord(BaseModel):
    """Represent one input request without altering its text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str

    @field_validator("prompt")
    @classmethod
    def check_nonblank(cls, value: str) -> str:
        """Reject blank requests while preserving original whitespace."""
        if not value.strip():
            raise ValueError("The prompt must not be blank.")
        return value


def load_prompts(path: Path) -> list[PromptRecord]:
    """Read and validate the complete prompt array."""
    adapter = TypeAdapter(list[PromptRecord])
    return adapter.validate_json(path.read_text(encoding="utf-8"))


def check_output_path(output: Path, inputs: list[Path]) -> None:
    """Reject directories, symlinks, and input/output collisions."""
    if output.is_symlink():
        raise ValueError("The output path must not be a symbolic link.")

    if output.exists() and not output.is_file():
        raise ValueError("The output path must refer to a regular file.")

    for source in inputs:
        if output.resolve() == source.resolve():
            raise ValueError("The output path must not overwrite an input.")

        # Detect hard links when both paths already exist.
        if output.exists() and source.exists():
            if output.samefile(source):
                raise ValueError(
                    "The output file is also an input file."
                )


def write_json_atomic(
    output: Path,
    records: list[dict[str, object]],
) -> None:
    """Serialize first, then replace the destination with a full file."""
    # Serialization failures must happen before touching the output.
    payload = json.dumps(
        records,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # The temporary file is closed before replacement.
        os.replace(temporary, output)
        temporary = None

    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"Warning: could not remove {temporary}: {exc}",
                    file=sys.stderr,
                )
