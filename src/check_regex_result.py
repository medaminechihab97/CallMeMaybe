"""Inspect the generated numeric-replacement pattern."""

import json
from pathlib import Path


def main() -> None:
    """Display whether the saved regex contains a backspace."""
    try:
        path = Path("data/output/function_calling_results.json")
        records = json.loads(path.read_text(encoding="utf-8"))

        # Ninth record in this particular test fixture.
        pattern = records[8]["parameters"]["regex"]

        print("Decoded regex:", repr(pattern))
        print("Contains backspace:", "\x08" in pattern)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Could not inspect the result: {exc}")


if __name__ == "__main__":
    main()
