"""Entry point for the call-me-maybe function calling tool.

Usage::

    uv run python -m src \\
        --functions_definition data/input/functions_definition.json \\
        --input data/input/function_calling_tests.json \\
        --output data/output/function_calling_results.json
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with input/output path attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into structured function calls."
        )
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
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the output results JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: load inputs, run pipeline, save outputs."""
    args = parse_args()

    # Lazy imports so errors surface clearly
    try:
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
    except ImportError:
        print(
            "Error: llm_sdk package not found. "
            "Copy the llm_sdk/ directory next to src/ and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.file_handler import (
        load_function_definitions,
        load_test_prompts,
        save_results,
    )
    from src.pipeline import run_pipeline

    print("Loading function definitions...", file=sys.stderr)
    functions = load_function_definitions(args.functions_definition)
    print(f"  Loaded {len(functions)} function(s).", file=sys.stderr)

    print("Loading test prompts...", file=sys.stderr)
    prompts = load_test_prompts(args.input)
    print(f"  Loaded {len(prompts)} prompt(s).", file=sys.stderr)

    print("Initializing LLM model...", file=sys.stderr)
    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"Error: Failed to initialize LLM model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Running pipeline...", file=sys.stderr)
    results = run_pipeline(model, prompts, functions)

    print(
        f"Saving {len(results)} result(s) to {args.output}...", file=sys.stderr
    )
    save_results(results, args.output)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
