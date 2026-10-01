"""Select one function and generate its constrained parameters."""

import json
import math
import sys
from pathlib import Path
from typing import cast

import numpy as np
from llm_sdk import Small_LLM_Model

from src.parameter_parser import (
    ParameterObjectParser,
    build_parameter_parser,
)
from src.select_function import (
    FunctionDefinition,
    load_functions,
    select_function,
)
from src.token_text import load_token_text


def choose_token(
    logits: list[float],
    fragments: dict[int, str],
    parser: ParameterObjectParser,
) -> tuple[int, str, ParameterObjectParser]:
    """Choose the highest-scoring token that the parser accepts."""
    scores = np.asarray(logits, dtype=np.float64)

    if scores.ndim != 1 or scores.size == 0:
        raise RuntimeError("Expected a nonempty vector of logits.")

    if not np.isfinite(scores).all():
        raise RuntimeError("The model returned non-finite logits.")

    for candidate in np.argsort(scores)[::-1]:
        token_id = int(candidate)
        text = fragments.get(token_id)

        # Missing tokens include special and non-ASCII tokens.
        if text is None:
            continue

        try:
            next_parser = parser.accept(text)
        except ValueError:
            continue

        return token_id, text, next_parser

    raise RuntimeError("No token can continue the parameter object.")


def validate_parameters(
    text: str,
    function: FunctionDefinition,
) -> dict[str, object]:
    """Recheck completed syntax, exact keys, types, and finite numbers."""
    parser = build_parameter_parser(function.parameters).accept(text)

    if not parser.is_complete():
        raise ValueError("The parameter object is incomplete.")

    value = json.loads(text)

    if not isinstance(value, dict):
        raise ValueError("Parameters must be an object.")

    if set(value) != set(function.parameters):
        raise ValueError("Parameter names do not match the definition.")

    for name, metadata in function.parameters.items():
        item = value[name]
        kind = metadata["type"]
        if kind == "integer":
            if type(item) is not int:
                raise ValueError(f"{name!r} must be an integer.")

        elif kind == "number":
            # Accept only actual numbers, not strings or booleans.
            if type(item) not in (int, float):
                raise ValueError(f"{name!r} must be a number.")

            # Convert integer-looking numbers: 2 becomes 2.0.
            try:
                number = float(item)
            except OverflowError as exc:
                raise ValueError(
                    f"{name!r} is too large for float output."
                ) from exc

            if not math.isfinite(number):
                raise ValueError(f"{name!r} must be finite.")

            # Store the float in the dictionary that will be saved.
            value[name] = number

        elif kind == "string":
            if type(item) is not str:
                raise ValueError(f"{name!r} must be a string.")

        elif kind == "boolean":
            if type(item) is not bool:
                raise ValueError(f"{name!r} must be a boolean.")

        else:
            raise ValueError(f"Unsupported parameter type: {kind!r}")

    return cast(dict[str, object], value)


def generate_parameters(
    model: Small_LLM_Model,
    function: FunctionDefinition,
    question: str,
    fragments: dict[int, str],
    max_tokens: int = 256,
) -> dict[str, object]:
    """Extract arguments while constraining each generated token."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    parser = build_parameter_parser(function.parameters)
    definition = json.dumps(function.model_dump(), ensure_ascii=True)
    order = json.dumps(list(function.parameters), ensure_ascii=True)

    instructions = (
        "Extract the arguments for the selected function. "
        "Do not execute the function or calculate its result. "
        "Return ONLY its parameters as a compact JSON object. "
        "Use every parameter exactly once in the specified order. "
        "Do not add whitespace outside string values. "
        "Use JSON Unicode escapes for non-ASCII characters. "
        "Extract values from the user's request.\n"
        f"Selected function: {definition}\n"
        f"Parameter order: {order}"
    )
    example_pattern = r"\bTOKEN\b"
    example_json = json.dumps({"pattern": example_pattern})

    instructions += (
        "\nString encoding rules:\n"
        "Preserve the intended string value after JSON decoding. "
        "A literal backslash in a string must be escaped in JSON. "
        "For regex arguments, distinguish regex escapes from JSON "
        "control-character escapes. Do not substitute a backspace "
        "character for a regex word boundary.\n"
        f"Encoding example only: the regex {example_pattern} "
        f"is represented by this JSON: {example_json}\n"
        "Do not copy TOKEN into your answer. Derive argument values "
        "from the user's request."
    )
    prompt = (
        f"<|im_start|>system\n{instructions}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []
    pieces: list[str] = []

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        token_id, text, parser = choose_token(
            logits, fragments, parser
        )

        input_ids.append(token_id)
        generated_ids.append(token_id)
        pieces.append(text)

        # Check the mapping immediately, not only after completion.
        expected = "".join(pieces)
        actual = model.decode(generated_ids)

        if actual != expected:
            raise RuntimeError(
                f"Token decoding mismatch at step {step + 1}.\n"
                f"Adapter text: {expected!r}\n"
                f"SDK text: {actual!r}"
            )
        if parser.is_complete():
            return validate_parameters(expected, function)

    partial = "".join(pieces)
    active_state = (
        parser.value.model_dump()
        if parser.value is not None
        else None
    )
    raise RuntimeError(
        "Parameter generation reached its limit before completion.\n"
        f"Selected function: {function.name}\n"
        f"Generated tokens: {len(generated_ids)}/{max_tokens}\n"
        f"Object part: {parser.part_index}/{len(parser.parts)}\n"
        f"Literal offset: {parser.literal_offset}\n"
        f"Value parser: {active_state!r}\n"
        f"Output beginning: {partial[:200]!r}\n"
        f"Output ending: {partial[-200:]!r}"
    )


def main() -> int:
    """Run the function-selection and parameter-extraction stages."""
    try:
        functions = load_functions(
            Path("data/input/functions_definition.json")
        )

        # Reject unsupported schemas before expensive model loading.
        for function in functions:
            build_parameter_parser(function.parameters)

        question = input("Your request: ")
        if not question.strip():
            raise ValueError("The request must not be empty.")

        print("Loading model...", file=sys.stderr, flush=True)
        model = Small_LLM_Model(device="cpu")
        fragments = load_token_text(model)

        print("Selecting function...", file=sys.stderr, flush=True)
        selection = json.loads(
            select_function(model, functions, question)
        )

        by_name = {function.name: function for function in functions}
        selected = by_name[selection["name"]]

        print(
            f"Extracting parameters for {selected.name}...",
            file=sys.stderr,
            flush=True,
        )

        parameters = generate_parameters(
            model, selected, question, fragments
        )

        result = {
            "prompt": question,
            "name": selected.name,
            "parameters": parameters,
        }

        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ))
        return 0

    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
