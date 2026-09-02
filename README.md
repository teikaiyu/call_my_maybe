*This project has been created as part of the 42 curriculum by <heychong>.*

# Project: call me maybe
Function Calling via Constrained Decoding

## Description
This project implements a function-calling system on top of a small (0.6B-parameter) local languate model, Qwen-3.06B, Given a natural-language prompt and a set of available function definitions, the system outputs a structured JSON object identifying which function should be called and with which arguments — instead of answering the question in free text.

The core technical challenge is reliability: small models are unreliable at producing well-formed, schema-compliant JSON when simply prompted to do so. This project solves that with **constrained decoding**: at every generation step, the model's output logits are restricted so that only tokens consistent with valid JSON and the target schema can be selected. This guarantees 100% syntactically valid, schema-compliant output, regardless of the model's own reliability.

## Instructions
Dependencies are managed with `uv`. From the project root:
```makefile
make install  #uv sync
make run  # runs with default input/output paths
make lint  # flake8 + mypy(standard)
make lint-strict  # flake8 + mypy --strict
make debug  # run under pdb
make clean  # remove caches and generated output(default)
```
To run with custom files:
```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_test.json \
    --output data/output/function_calling_results.json
```
All three arguments are optional: each falls back to the default path shown above if omitted. Arguments may be given in any order.

`make run` also accepts overrides via three Makefile variables —`FN_DEF`, `INPUT`, and `OUTPUT`— which map to `--functions_definition`, `--input`, and `--output` respectively (shortened for convenience at the call site). For example:
```
make run FN_DEF=data/input/custom_functions.json
```

### Algorithm Explanation
Generation proceeds token by token. At each step:
1. The current token sequence is passed to the model, which returns raw logits (unnormalized scores) for every token in the vocabulary.
2. Every candidate token is decoded back to text (using a byte-level BPE decode table built from `vocab.json`) and checked against the current constraint: is this token a valid continuation of the JSON structure and the active schema field?
3. Invalid tokens are excluded from consideration; the highest-scoring remaining token is selected (equivalent to masking invalid tokens to `-inf` before argmax).
4. The chosen token is appended to the sequence, and the loop repeats.

The JSON structure itself is treated as a sequence of typed regions, each handled by a dedicated generation routine:
- **Literal regions** (`{"prompt": "`, key names, punctuation):the model is constrained to reproduce an exact known string, one token at a time.
- **Enum regions** (the `name` field): constrained to exactly match one of the function names declared in `functions_definition.json`.
- **Typed value regions** (`parameters`): constrained according to the declared type ーfree-form quoted text for `string`, digit/`.`/`-` only for `number`, or `true`/`false` for `boolean`.

Because the function set and parameter schema are read from `functions_definition.json` at runtime, no part of the schema is hard-coded; the constraint sets are built dynamically for each function.

### Design Decisions
- **Literal text is emitted via `encode()` + direct concatenation**, not through the token-by-token masking machinery used for uncertain fields. The content is already fully known, so masking would only add vocabulary scans without changing the outcome; this materially improves speed given how slow inference is on this model/hardware. This is a deliberate application of constrained decoding; the technique's purpose is to bound the model's choices at points where the model is actually making a decision (which function to call, what value to assign a parameter). Structural tokens (braces, colons, key names, whitespace) are not decisions — they are fixed by the schema and JSON syntax regardless of the prompt, Applying full logit-masking machinery to spans with exactly one valid continuation would not change what gets emitted; it would only repeat the same vocabulary scan for no behavoiral difference. The actual guarantee this project makes — reduced model uncertainty (JSON is not what the model reliably produces on its own) to zero, everywhere it matters — is fully present: `_generate_enum` and the per-type value generators (`_generate_number`, `_generate_free_string`, `_generate_enum_bare`) are exactly where the model's genuine uncertainly lives, and that is exactly where masking is applied.
- **`FunctionsDefinition`/`TestPrompt` are pydantic models** with `extra="forbid"`, so both missing and unexpected keys are rejected during parsing rather than surfacing later as a generation-time failure.
- **File-level errors (`FileReadError`) and schema-level errors (`SchemaValidationError`) are distinct exception types**, so callers can report which kind of problem occurred rather than a single generic failure.
- **A single failed prompt stops the whole run** rather than being skipped, with a clear error message identifying the failing prompt. This was a deliberate simplicity/robustness trade-off for this project's scope.
- **Only `string`, `number`, and `boolean` parameter types are supported**. `interger`, `array`, `object`, and `null` are not handled; the deeper nested-argument case is listed as a bonus item and was left out of scope.

### Performance Analysis
- **Reliability**: because invalid tokens are structurally excluded rather than merely discouraged, output is 100% valid JSOn and 100% schema-compliant by construction — This is guaranteed by the decoding algorithm itself, not by the model's behavior.
- **Speed**: the dominant cost is the per-step scan over the vocabulary to evaluate which tokens are valid continuations. Literal (known) spans skip this scan entirely; only uncertain fields (function name, parameter values) pay the full per-token vocabulary scan cost.
- **Function/argument selection accuracy** depends on the underlying model, not on the decoding mechanism — constrained decoding guarantees *valid* output, not necessarily *correct* output.

### Challenges Fased
- Reconcilling `Small_LLM_Model.encode()`(returns a 2-D `torch.Tensor`) with `get_logits_from_input_ids()` (expects a flat `list[int]`) required explicit conversion at each call site.
- Decoding raw vocabulary entries back into real text required implementing the standard GPT-style byte-level BPE reverse mapping, since the SDK only exposes the vocabulary file path, not a decode utility for arbitary candidate tokens.
- `mypy --sytrict` initially failed inside the vendored `llm_sdk` package (which lack a `py.typed` marker and is not something this project is allowed to modify) and inside relative-import resolution for the `src` package. Both were resolved via `[tool.mypy]` overrides (`follow_import = "skip"` for `llm_sdk.*`) and by adding an `__init__.py` to `src` so mypy could resolve `src` as an explicit package base.
- `flake8` initially crashed with a `RecursionError` while scanning `.venv`; resolved by excluding `.venv` and `data` via a `.flake8` config file.

### Testing Strategy
The implementation was verified through the full pipeline (`make install` → `make lint-strict` → `make run` → `make clean`) using the sample `data/input/` files, confirming the process completes without raising an exception and produces a valid `data/output/` file matching the required schema. Edge cases such as empty strings, unusually large numbers, and special characters have not been exhaustively tested and remain a known gap.

### Example Usage
```
uv run python -m src
```
Given `functions_definition.json` containing `fn_add_numbers` and `fn_reverse_string`, and `function_calling_tests.json` containing `"What is the sum of 2 and 3?"`, the output file will contain:
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    }
]
```

## Resources
- Hugging Face Transformers documentation
- pydantic documentation
- Byte Pair Encoding (BPE): Sennrich et al., *Neural MachineTranslation of Rare Words with Subword Units* (2016)
- GPT-3's byte-level BPE reverse-mapping scheme (public, widely reused algorithm for reconstructing text from byte-level BPE vocabulary entries)

### AI usage
An AI assistant (Claude) was used throughout this project in a question-based capacity: clarifying concepts (tensors, tokenization, BPE, constrained decoding theory), reviewing hand-written code for bugs and design issues, and discussing trade-offs (e.g. literal-span handling, exception granularity, mypy/flake8 configuration). All source code in `src/` was written by the author; the assistant did not author the final implementation but was consulted for debugging specific errors (e.g. the `mypy --strict` failures against the vendored SDK) and for two short illustrative code snippets early exploration, which were later superseded by the author's own implementation.
