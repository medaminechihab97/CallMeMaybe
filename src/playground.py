"""Explore next-token generation through the supplied SDK."""

import sys

import numpy as np

from llm_sdk import Small_LLM_Model


def generate_reply(model: Small_LLM_Model, question: str) -> str:
    """Generate a short reply using greedy next-token selection."""
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer briefly."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    # encode() returns a batch containing one sequence.
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []

    # Obtain stop-token IDs through the public encode() method.
    stop_ids: set[int] = set()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        marker_ids: list[int] = model.encode(marker).tolist()[0]
        if len(marker_ids) != 1:
            raise ValueError(f"Expected one token for {marker!r}")
        stop_ids.add(marker_ids[0])

    for _ in range(64):
        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        next_id = int(np.argmax(logits))

        if next_id in stop_ids:
            break

        input_ids.append(next_id)
        generated_ids.append(next_id)
    else:
        print("[Stopped at the 64-token limit.]", file=sys.stderr)

    return str(model.decode(generated_ids))


def main() -> int:
    """Load the model and run a simple interactive prompt."""
    try:
        print("Loading Qwen3-0.6B on CPU...", flush=True)
        model = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B",
            device="cpu",
        )
        print("Ready. Type 'quit' to exit.")

        while True:
            question = input("\nYou: ").strip()

            if question.lower() in {"quit", "exit"}:
                return 0
            if not question:
                continue

            print("Generating...", flush=True)
            reply = generate_reply(model, question)
            print(f"Model: {reply}")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        return 0
    except Exception as exc:
        # Broad handling at the CLI boundary makes setup failures visible.
        print(
            f"Error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())