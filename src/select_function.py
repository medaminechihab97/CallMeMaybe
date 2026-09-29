"""Select a function using constrained token generation."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, TypeAdapter

from src.prefix_parser import PrefixParser


class FunctionDefinition(BaseModel):
    """Validate the outer structure of a function definition."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, Any]


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Read definitions and reject empty or duplicate function names."""
    adapter = TypeAdapter(list[FunctionDefinition])
    functions = adapter.validate_json(path.read_text(encoding="utf-8"))

    if not functions:
        raise ValueError("The function definition list is empty.")

    names = [function.name for function in functions]

    if any(not name.strip() for name in names):
        raise ValueError("Function names must not be empty.")

    if len(names) != len(set(names)):
        raise ValueError("Function names must be unique.")

    return functions


def select_function(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    question: str,
) -> str:
    """Generate one allowed function-selection JSON object."""
    candidates = [
        json.dumps({"name": function.name}, ensure_ascii=True)
        for function in functions
    ]

    sequences: list[list[int]] = [
        model.encode(candidate).tolist()[0]
        for candidate in candidates
    ]

    if any(not sequence for sequence in sequences):
        raise ValueError("A candidate encoded to an empty sequence.")

    parser = PrefixParser(remaining=sequences)

    definitions = json.dumps(
        [function.model_dump() for function in functions],
        ensure_ascii=True,
    )

    instructions = (
        "Select the function that best matches the user's request. "
        "Do not execute it or answer the request. "
        'Output only a JSON object with one key: "name".\n'
        f"Available functions:\n{definitions}"
    )

    # Same Qwen3-specific prompt format as the earlier playground.
    prompt = (
        f"<|im_start|>system\n{instructions}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []

    # Each accepted token shortens the remaining candidates.
    max_steps = max(len(sequence) for sequence in sequences)

    for _ in range(max_steps):
        scores = np.asarray(
            model.get_logits_from_input_ids(input_ids),
            dtype=np.float64,
        )

        allowed = sorted(parser.allowed_token_ids())

        if not allowed:
            raise RuntimeError("No valid next token is available.")

        if min(allowed) < 0 or max(allowed) >= scores.size:
            raise RuntimeError("Token IDs do not match the logits.")

        if not np.isfinite(scores[allowed]).all():
            raise RuntimeError("The model returned non-finite scores.")

        masked_scores = np.full(scores.shape, -np.inf)
        masked_scores[allowed] = scores[allowed]

        next_id = int(np.argmax(masked_scores))

        parser = parser.accept(next_id)
        input_ids.append(next_id)
        generated_ids.append(next_id)

        if parser.is_complete():
            result = str(model.decode(generated_ids))

            # Check the actual decoded text, not just the token IDs.
            if result not in candidates:
                raise RuntimeError("Decoded output is not a candidate.")

            return result

    raise RuntimeError("Generation ended before a candidate completed.")


def main() -> int:
    """Load definitions and the model, then select one function."""
    try:
        path = Path("data/input/functions_definition.json")
        functions = load_functions(path)

        question = input("Your request: ").strip()

        if not question:
            raise ValueError("The request must not be empty.")

        print("Loading model...", file=sys.stderr, flush=True)
        model = Small_LLM_Model(device="cpu")

        result = select_function(model, functions, question)
        print(result)
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