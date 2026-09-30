"""Build a printable-ASCII token table for a byte-level tokenizer."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter


def byte_symbol_decoder() -> dict[str, int]:
    """Invert the standard byte-level tokenizer symbol mapping."""
    direct = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 173))
        + list(range(174, 256))
    )

    byte_values = direct.copy()
    codepoints = direct.copy()
    extra = 0

    for byte in range(256):
        if byte not in direct:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1

    return {
        chr(codepoint): byte
        for codepoint, byte in zip(codepoints, byte_values)
    }


def load_token_text(model: Small_LLM_Model) -> dict[int, str]:
    """Read ordinary vocabulary tokens and retain printable ASCII."""
    path = Path(model.get_path_to_tokenizer_file())
    metadata = json.loads(path.read_text(encoding="utf-8"))

    if metadata.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("This adapter requires a ByteLevel decoder.")

    if metadata.get("model", {}).get("type") != "BPE":
        raise ValueError("This adapter requires a BPE vocabulary.")

    vocabulary = TypeAdapter(dict[str, int]).validate_python(
        metadata["model"]["vocab"],
        strict=True,
    )

    # Exclude added/special tokens even if also present in the model vocab.
    added_ids = {
        entry["id"]
        for entry in metadata.get("added_tokens", [])
    }

    decoder = byte_symbol_decoder()
    fragments: dict[int, str] = {}

    for spelling, token_id in vocabulary.items():
        if token_id in added_ids:
            continue

        try:
            raw = bytes(decoder[symbol] for symbol in spelling)
        except KeyError as exc:
            raise ValueError(
                "Vocabulary does not match the byte-level mapping."
            ) from exc

        if raw and all(
            0x20 <= byte <= 0x7E or byte in (9, 10, 13)
            for byte in raw
            ):
            fragments[token_id] = raw.decode("ascii")

    # Ensure basic compact JSON text can be represented.
    single_characters = {
        text for text in fragments.values() if len(text) == 1
    }
    required = {chr(code) for code in range(0x20, 0x7F)}

    if not required.issubset(single_characters):
        raise ValueError("Tokenizer lacks printable ASCII fallback tokens.")

    return fragments