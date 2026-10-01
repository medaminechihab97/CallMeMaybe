"""Check the numeric regex in the current demonstration fixture."""

import json
import re
from pathlib import Path


def main() -> None:
    """Compare generated matches with the fixture's expected matches."""
    try:
        path = Path("data/output/function_calling_results.json")
        records = json.loads(path.read_text(encoding="utf-8"))
        parameters = records[8]["parameters"]

        pattern = parameters["regex"]
        source = parameters["source_string"]

        matches = [
            match.group(0)
            for match in re.finditer(pattern, source)
        ]

        print("Decoded regex:", repr(pattern))
        print("Matches:", matches)

        if matches != ["34", "233"]:
            raise ValueError("Expected matches ['34', '233'].")

        print("Numeric matching test passed.")

    except (OSError, ValueError, KeyError, IndexError, TypeError,
            re.error) as exc:
        raise SystemExit(f"Test failed: {exc}")


if __name__ == "__main__":
    main()
