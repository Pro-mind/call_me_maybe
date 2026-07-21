"""Handles loading and saving of JSON input/output files."""

import json
import sys
from pathlib import Path
from typing import List

from src.models import FunctionCallResult, FunctionDefinition


def load_function_definitions(path: Path) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        List of validated FunctionDefinition objects.
    """
    if not path.exists():
        print(f"Error: Function definitions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in function definitions file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw, list):
        print("Error: Function definitions file must contain a JSON array.", file=sys.stderr)
        sys.exit(1)

    definitions: List[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            definitions.append(FunctionDefinition(**item))
        except Exception as e:
            print(f"Error: Invalid function definition at index {i}: {e}", file=sys.stderr)
            sys.exit(1)
    
    unknown_fn = FunctionDefinition(
        name="fn_unknown",
        description=(
            "Use this function for no content or meaningless, random, incomplete, or "
            "unsupported user input that does not clearly describe any "
            "available operation or not word like dsjyhgv."
        ),
        parameters={
        },
        returns={
            "type": "string"
        },
    )

    definitions.append(unknown_fn)

    return definitions


def load_test_prompts(path: Path) -> List[str]:
    """Load natural language prompts from a JSON file.

    Args:
        path: Path to the function_calling_tests.json file.

    Returns:
        List of prompt strings.
    """
    if not path.exists():
        print(f"Error: Test prompts file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in test prompts file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw, list):
        print("Error: Test prompts file must contain a JSON array.", file=sys.stderr)
        sys.exit(1)

    prompts: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "prompt" not in item:
            print(f"Error: Missing 'prompt' key in test entry at index {i}.", file=sys.stderr)
            sys.exit(1)
        prompts.append(item["prompt"])

    return prompts


def save_results(results: List[FunctionCallResult], path: Path) -> None:
    """Save function call results to a JSON output file.

    Args:
        results: List of FunctionCallResult objects to serialize.
        path: Destination path for the output JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    output = [r.model_dump() for r in results]

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Error: Could not write output file: {e}", file=sys.stderr)
        sys.exit(1)
