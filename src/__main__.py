"""Run batch function calling with python -m src."""

import argparse
import json
import sys
from pathlib import Path

from src.batch_io import (
    check_output_path,
    load_prompts,
    write_json_atomic,
)


def build_cli() -> argparse.ArgumentParser:
    """Define the assignment's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert user prompts into constrained function calls."
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the JSON array of prompt records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Destination for the complete results JSON array.",
    )

    return parser


def run_batch(
    definitions_path: Path,
    input_path: Path,
    output_path: Path,
) -> int:
    """Generate all results and publish them only after full success."""
    check_output_path(
        output_path,
        [definitions_path, input_path],
    )
    prompts = load_prompts(input_path)

    # Delayed imports let --help run without importing the model SDK.
    from src.call_once import generate_parameters, validate_parameters
    from src.parameter_parser import build_parameter_parser
    from src.select_function import load_functions, select_function

    functions = load_functions(definitions_path)

    # Reject unsupported parameter schemas before loading weights.
    for function in functions:
        build_parameter_parser(function.parameters)

    results: list[dict[str, object]] = []

    # An empty input array produces [] without loading model weights.
    if prompts:
        from llm_sdk import Small_LLM_Model

        from src.token_text import load_token_text

        print("Loading model...", file=sys.stderr, flush=True)
        model = Small_LLM_Model(device="cpu")
        fragments = load_token_text(model)

        by_name = {
            function.name: function
            for function in functions
        }
        total = len(prompts)

        for index, record in enumerate(prompts, start=1):
            print(
                f"[{index}/{total}] Processing prompt...",
                file=sys.stderr,
                flush=True,
            )

            try:
                selection = json.loads(
                    select_function(model, functions, record.prompt)
                )

                if not isinstance(selection, dict):
                    raise ValueError("Function selection is not an object.")

                if set(selection) != {"name"}:
                    raise ValueError("Unexpected function-selection keys.")

                name = selection["name"]

                if not isinstance(name, str) or name not in by_name:
                    raise ValueError("The selected function is unknown.")

                selected = by_name[name]

                parameters = generate_parameters(
                    model,
                    selected,
                    record.prompt,
                    fragments,
                )

                # Recheck the object that will actually be saved.
                parameters = validate_parameters(
                    json.dumps(
                        parameters,
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                    selected,
                )

                results.append({
                    "prompt": record.prompt,
                    "name": selected.name,
                    "parameters": parameters,
                })

            except Exception as exc:
                raise RuntimeError(
                    f"Prompt {index}/{total} failed "
                    f"({type(exc).__name__}): {exc}"
                ) from exc

    # Repeat the collision check immediately before publishing.
    check_output_path(
        output_path,
        [definitions_path, input_path],
    )
    write_json_atomic(output_path, results)

    return len(results)


def main() -> int:
    """Parse arguments and report batch failures without a traceback."""
    args = build_cli().parse_args()

    try:
        count = run_batch(
            args.functions_definition,
            args.input,
            args.output,
        )
    except KeyboardInterrupt:
        print(
            "\nCancelled. Check the output timestamp before reusing "
            "an existing results file.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Error ({type(exc).__name__}): {exc}\n"
            "The batch did not complete successfully. An existing "
            "results file may belong to an earlier run.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Saved {count} result(s) to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
