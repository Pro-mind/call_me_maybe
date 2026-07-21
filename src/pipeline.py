"""Pipeline that processes all prompts end-to-end."""

import sys
from typing import Any, List

from src.constrained_decoder import (
    generate_function_call,
    load_vocab,
    select_function,
)
from src.models import FunctionCallResult, FunctionDefinition
from src.tokenizer import BPETokenizer


def run_pipeline(
    model: Any,
    prompts: List[str],
    functions: List[FunctionDefinition],
) -> List[FunctionCallResult]:
    """Process all prompts and return structured function call results.

    For each prompt:
      1. Use the LLM + constrained decoding to select the best function.
      2. Use the LLM + constrained decoding to generate typed arguments.
      3. Wrap the result in a FunctionCallResult.

    Args:
        model:     The Small_LLM_Model instance.
        prompts:   List of natural language prompt strings.
        functions: List of available FunctionDefinition objects.

    Returns:
        List of FunctionCallResult objects ready for serialization.
    """
    vocab_path = model.get_path_to_vocab_file()
    id_to_token = load_vocab(vocab_path)

    from src.constrained_decoder import _flatten_encode

    # Default: model's own tokenizer (correct IDs guaranteed)
    encode_fn = lambda text: _flatten_encode(model, text)  # noqa: E731

    # Bonus: validate custom BPE tokenizer, upgrade if IDs match
    try:
        merges_path = model.get_path_to_merges_file()
        tokenizer = BPETokenizer(vocab_path, merges_path)
        test_text = "hello world 123"
        if tokenizer.encode(test_text) == _flatten_encode(model, test_text):
            encode_fn = tokenizer.encode
            print("Using validated custom BPE tokenizer.", file=sys.stderr)
        else:
            print("BPE tokenizer IDs differ — using model.encode().", file=sys.stderr)
    except Exception as e:
        print(f"BPE tokenizer unavailable ({e}).", file=sys.stderr)

    results: List[FunctionCallResult] = []

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{len(prompts)}] Processing: {prompt}", file=sys.stderr)

        try:
            selected_fn = select_function(model, prompt, functions, encode_fn)
            print(f"  -> Selected function: {selected_fn.name}", file=sys.stderr)

            parameters = generate_function_call(
                model, prompt, selected_fn, id_to_token, encode_fn
            )
            print(f"  -> Parameters: {parameters}", file=sys.stderr)

            results.append(FunctionCallResult(
                prompt=prompt,
                name=selected_fn.name,
                parameters=parameters,
            ))

        except Exception as e:
            print(f"  Error processing prompt '{prompt}': {e}", file=sys.stderr)
            if functions:
                results.append(FunctionCallResult(
                    prompt=prompt,
                    name=functions[0].name,
                    parameters={},
                ))

    return results
