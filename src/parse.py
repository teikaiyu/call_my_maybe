"""Parse and validate the two input files into typed, in-memory objects.

- functions_definition.json -> list[FunctionDefinition]
- function_calling_tests.json -> list[TestPrompt]

File-level errors (missing file, malformed JSON) and shema-level errors
(missing/extra keys, wrong types) are raised as distinct exception types so
callers can report them differently, per project requirements.
"""

import json
from typing import Any, Literal
from pathlib import Path
from pydantic import BaseModel, ConfigDict, ValidationError


class FileReadError(Exception):
    """Raised with the file is missing or is not valid JSON."""


class SchemaValidationError(Exception):
    """Raised when the JSON is valid but doesn't match the expected schema."""


class ParameterSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean"]


class ReturnSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ReturnSchema


class TestPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


def _load_json(path: Path) -> Any:
    """Read and JSON-decode a file. Raises FileReadError on any I/O or
    syntax problem, with a message identifying which file failed."""
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileReadError(f"file not found: {path}") from e
    except UnicodeDecodeError as e:
        raise FileReadError(f"could not decode {path} as UTF-8") from e
    except json.JSONDecodeError as e:
        raise FileReadError(f"invalid JSON in {path}: {e}") from e


def parse_functions_definition(path: Path) -> list[FunctionDefinition]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise SchemaValidationError(f"{path}: expected a JSON array")
    functions: list[FunctionDefinition] = []
    for i, entry in enumerate(raw):
        try:
            functions.append(FunctionDefinition.model_validate(entry))
        except ValidationError as e:
            raise SchemaValidationError(
                f"{path}: entry {i} does not match expected schema: {e}"
            ) from e
    return functions


def parse_test_prompts(path: Path) -> list[TestPrompt]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise SchemaValidationError(
            f"{path}: expected a JSON array at top level"
        )
    prompts: list[TestPrompt] = []
    for i, entry in enumerate(raw):
        try:
            prompts.append(TestPrompt.model_validate(entry))
        except ValidationError as e:
            raise SchemaValidationError(
                f"{path}: entry {i} does not match expected schema: {e}"
            ) from e
    return prompts
