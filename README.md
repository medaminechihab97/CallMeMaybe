*This project has been created as part of the 42 curriculum by mochihab.*

# CallMeMaybe

## Description
CallMeMaybe translates natural-language requests into structured function
calls using **Qwen/Qwen3-0.6B** and constrained decoding.
It selects a function and extracts arguments without executing the function.
Model interaction uses only the supplied SDK's public interface.

## Instructions
Requirements: Python 3.10+, `uv`, the supplied `llm_sdk/`, and optionally
`make`. Initial dependency and model downloads require internet access.
Run all commands from the repository root.

```bash
uv sync
uv run python -m src
```

Default inputs:
- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

Default output: `data/output/function_calling_results.json`.

### Custom paths
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/edge_cases.json \
  --output data/output/edge_case_results.json
```

### Development
```bash
make install  # Install dependencies
make run      # Run the default batch
make debug    # Start Python's debugger
make lint     # Run Flake8 and mypy
make clean    # Remove project caches
```

## Example Usage
For “What is the sum of 2 and 3?”, the desired result is:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```
Verify float normalization in the final implementation; earlier versions
serialized whole-valued `number` arguments as integers.

## Algorithm Explanation
1. Validate requests and function definitions before loading model weights.
2. Tokenize candidate function-name objects and track their valid prefixes.
3. Mask invalid selection tokens and choose the highest-scoring valid token.
4. Compile the selected function's parameters into a state-based JSON parser.
5. Test candidate token fragments against that parser, accepting the
   highest-scoring valid continuation without mutating rejected trial states.
6. Stop at object completion, verify decoding, and validate keys and types.
7. Publish results through a temporary file only after the full batch succeeds.

## Design Decisions
- Separate function selection from argument extraction.
- Use greedy constrained decoding rather than repairing malformed JSON.
- Generate required parameters in definition order.
- Use Pydantic models for validation and immutable parser states.
- Represent Unicode through JSON escapes in the ASCII token adapter.
- Reject unsupported schemas; nested objects and arrays are not supported.
- Keep the supplied SDK unchanged and document dependency-check workarounds.

## Performance Analysis
The original 11-call development batch appeared correct by inspection.
An eight-case edge fixture selected every function correctly, but only
four calls had fully correct arguments. These are historical fixture results,
not general accuracy estimates. Runtime and the 90% accuracy target remain
unverified. Valid structure does not guarantee correct meaning or completion.

## Challenges Faced
Development addressed regex/JSON escaping, multi-character token boundaries,
integer parsing, validation indentation, and SDK type-resolution issues.
Exact preservation of empty strings, punctuation, Unicode, and whole-word
regex semantics remains an important regression-testing area.

## Testing Strategy
Development scripts exercised parser states, split fragments, whitespace,
integer restrictions, token rejection, and batch I/O.
Rerun original and edge-case fixtures after changes; inspect values and regex
behavior, not just JSON validity. Test malformed inputs and failed writes.
Retain local regression tests, measure runtime, and do not commit outputs.

## Resources
- [Python JSON](https://docs.python.org/3/library/json.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [uv](https://docs.astral.sh/uv/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [mypy](https://mypy.readthedocs.io/en/stable/)
- [Flake8](https://flake8.pycqa.org/en/latest/)

### AI Assistance
help with understanding basic concepts, discuss some design decisions and help with README drafting.
