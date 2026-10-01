"""Test batch input validation and safe output publication."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.batch_io import (
    check_output_path,
    load_prompts,
    write_json_atomic,
)


def main() -> None:
    """Verify successful writes and preservation after failures."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "prompts.json"
        output = root / "results.json"

        source.write_text(
            '[{"prompt":"  Greet Ali  "}]',
            encoding="utf-8",
        )
        prompts = load_prompts(source)
        assert prompts[0].prompt == "  Greet Ali  "

        output.write_text('[{"old":true}]\n', encoding="utf-8")
        original = output.read_bytes()

        records: list[dict[str, object]] = [{
            "prompt": "Greet Ali",
            "name": "fn_greet",
            "parameters": {"name": "Ali"},
        }]

        # A failed replacement must preserve the old destination.
        with patch(
            "src.batch_io.os.replace",
            side_effect=OSError("Simulated replacement failure"),
        ):
            try:
                write_json_atomic(output, records)
            except OSError:
                pass
            else:
                raise AssertionError("Expected replacement failure.")

        assert output.read_bytes() == original
        assert not list(root.glob(".results.json.*.tmp"))

        # Serialization failure must also preserve the old file.
        try:
            write_json_atomic(output, [{"value": float("nan")}])
        except ValueError:
            pass
        else:
            raise AssertionError("Expected non-finite number rejection.")

        assert output.read_bytes() == original

        # Successful publication replaces the old results.
        write_json_atomic(output, records)
        assert json.loads(output.read_text(encoding="utf-8")) == records

        # Refuse to overwrite an input.
        try:
            check_output_path(source, [source])
        except ValueError:
            pass
        else:
            raise AssertionError("Expected input/output collision.")

        for invalid in (
            "not JSON",
            "{}",
            '[{"prompt":123}]',
            '[{"prompt":"   "}]',
        ):
            source.write_text(invalid, encoding="utf-8")
            try:
                load_prompts(source)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted input: {invalid}")

        # Empty arrays are supported.
        source.write_text("[]", encoding="utf-8")
        assert load_prompts(source) == []

        nested = root / "new" / "empty.json"
        write_json_atomic(nested, [])
        assert json.loads(nested.read_text(encoding="utf-8")) == []

    print("Batch I/O tests passed.")


if __name__ == "__main__":
    main()
