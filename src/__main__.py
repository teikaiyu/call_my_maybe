import sys
import json
from pathlib import Path
from .parse import (
    FileReadError, SchemaValidationError,
    parse_functions_definition, parse_test_prompts
)
from .constrained_decode import ConstrainedJSONGenerator, load_vocab
from llm_sdk import Small_LLM_Model


DEFAULTS: dict[str, str] = {
    "--functions_definition": "data/input/functions_definition.json",
    "--input": "data/input/function_calling_tests.json",
    "--output": "data/output/function_calling_results.json"
}


class ArgumentError(Exception):
    """Raised for any malformed command-lines invocation."""


def parse_args(argv: list[str]) -> dict[str, str]:
    """Parse argv (excluding the program name) into a dict with the three
    keys in DEFAULTS, falling back to defaults for any flag not given."""
    options: dict[str, str] = {}
    remaining = list(argv)
    while remaining:
        flag = remaining.pop(0)
        if flag not in DEFAULTS.keys():
            raise ArgumentError(f"unknown argument: {flag}")
        key = flag
        if key in options:
            raise ArgumentError(f"duplicate argument: {flag}")
        if not remaining:
            raise ArgumentError(f"{flag} requires a value")
        value = remaining.pop(0)
        if value.startswith("--"):
            raise ArgumentError(f"{flag} requires a value, got flag {value}")
        options[key] = value
    return {**DEFAULTS, **options}


def main() -> None:
    try:
        system = parse_args(sys.argv[1:])

        functions = parse_functions_definition(
            system["--functions_definition"])
        functions_raw = [f.model_dump() for f in functions]
        prompts = parse_test_prompts(system["--input"])

        model = Small_LLM_Model()
        vocab = load_vocab(model.get_path_to_vocab_file())
        results = []
        for p in prompts:
            gen = ConstrainedJSONGenerator(
                model, vocab, functions_raw, p.prompt)
            results.append(gen.generate())

        Path(system["--output"]).parent.mkdir(parents=True, exist_ok=True)
        with open(system["--output"], mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except ArgumentError as e:
        print(f"ArgumentError: {e}", file=sys.stderr)
        sys.exit(1)
    except FileReadError as e:
        print(f"FileReadError: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"OSError: {e}", file=sys.stderr)
        sys.exit(1)
    except SchemaValidationError as e:
        print(f"SchemaValidationError: {e}", file=sys.stderr)
        sys.exit(1)
    except MemoryError as e:
        print(f"MemoryError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
