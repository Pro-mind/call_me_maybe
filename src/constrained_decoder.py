"""Constrained decoding engine.

Two phases:
  1. Function selection  - the model picks among the known function names by
     scoring how probable the user prompt is under each function's context.
  2. Argument generation - parameters are filled by continuous JSON
     completion: we write every structural character of the JSON ourselves
     (braces, keys, quotes, commas) and let the model fill only the values.

Argument generation now implements *real* constrained decoding, following
the algorithm described in the subject (section V.3.3):

  1. The model produces logits for every token in the vocabulary.
  2. We determine, for the JSON schema type currently being generated,
     which tokens would keep the value both syntactically valid *and*
     schema-compliant (e.g. only digits/'-'/'.' for a number, only
     printable non-quote characters for a string, only "true"/"false"
     prefixes for a boolean).
  3. Every other token's logit is set to negative infinity.
  4. The next token is sampled (here: argmax) only among the remaining
     valid tokens.

This guarantees that every value we hand back is already valid for its
declared type -- we never rely on the model "spontaneously" producing
correct JSON, and we never patch up an invalid string after the fact.
"""

import math
from typing import Any, Callable, Dict, List, Set, Tuple

from src.models import FunctionDefinition

NEG_INF: float = -math.inf

# Characters a JSON number/integer literal is allowed to be made of. A
# leading space is tolerated because the prompt we build always ends with
# ": " right before the value, so a stray leading-space token from the
# tokenizer must not be rejected outright.
NUMBER_CHARS: str = "-0123456789. "

# Characters that can appear in a token that only ever spells out (part of)
# the words "true" or "false".
BOOLEAN_CHARS: Set[str] = set("truefals ")
BOOLEAN_LITERALS: Tuple[str, str] = ("true", "false")


def load_vocab(vocab_path: str) -> Dict[int, str]:
    """Load the tokenizer vocabulary file and invert it to id->token mapping.

    Args:
        vocab_path: Path returned by ``model.get_path_to_vocab_file()``.

    Returns:
        Dict mapping token_id -> token_string.
    """
    import json

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json: Dict[str, Any] = json.load(f)

    return {int(tid): tok for tok, tid in vocab_json.items()}


def token_to_text(token_str: str) -> str:
    """Convert a raw vocabulary token to its plain-text representation.

    Qwen / GPT-style tokenizers encode a leading space as 'G-dot' (U+0120)
    and a newline as 'C-dot' (U+010A).

    Args:
        token_str: Raw token string from the vocabulary file.

    Returns:
        Human-readable string for the token.
    """
    return token_str.replace("\u0120", " ").replace("\u010a", "\n")


def _flatten_encode(model: Any, text: str) -> List[int]:
    """Encode *text* using the SDK model and return a flat list of token ids.

    Handles both 1-D and batched 2-D outputs from model.encode().

    Args:
        model: The Small_LLM_Model instance.
        text:  Text to encode.

    Returns:
        Flat list of integer token ids.
    """
    encoded: List[Any] = model.encode(text).tolist()
    if encoded and isinstance(encoded[0], list):
        return list(encoded[0])
    return [int(x) for x in encoded]


def top_token_id(prompt_ids: List[int], model: Any) -> int:
    """Return the single highest-scoring next-token id for a prompt.

    Args:
        prompt_ids: Token ids of the prompt so far.
        model:      The Small_LLM_Model instance.

    Returns:
        The token id with the maximum logit.
    """
    logits = model.get_logits_from_input_ids(prompt_ids)
    return logits.index(max(logits))


def parse_numb(value_str: str) -> float:
    """Parse an accumulated number string into a float, safely.

    Constrained decoding should already guarantee ``value_str`` is a
    well-formed number, but this stays as a defensive net so a
    pathological edge case (e.g. hitting the token cap before a digit was
    ever produced) can never crash the program.

    Args:
        value_str: The raw number text collected during decoding.

    Returns:
        The parsed float, or 0.0 if empty or malformed.
    """
    cleaned = value_str.strip()
    if cleaned.endswith(","):
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------- #
# Grammar helpers: decide, character by character, whether a value-so-far
# is still a valid *prefix* of the target JSON type, and when it becomes a
# valid, complete value of that type.
# --------------------------------------------------------------------- #


def _number_prefix_valid(text: str, allow_float: bool) -> bool:
    """Check whether *text* could still grow into a valid JSON number.

    Args:
        text:        The value accumulated so far.
        allow_float: Whether a single decimal point is permitted
            (``True`` for "number", ``False`` for "integer").

    Returns:
        True if *text* is a valid (possibly incomplete) numeric prefix.
    """
    text = text.lstrip(" ")
    if text == "":
        return True
    if not allow_float and "." in text:
        return False

    seen_dot = False
    for i, ch in enumerate(text):
        if ch == "-":
            if i != 0:
                return False
        elif ch == ".":
            if not allow_float or seen_dot or i == 0 or not text[i - 1].isdigit():
                return False
            seen_dot = True
        elif ch.isdigit():
            continue
        else:
            return False
    return True


def _number_is_complete(text: str, allow_float: bool) -> bool:
    """Check whether *text* is already a fully valid JSON number.

    Args:
        text:        The value accumulated so far.
        allow_float: Whether a decimal point is permitted.

    Returns:
        True if *text* parses as a complete number/integer literal.
    """
    text = text.lstrip(" ")
    if text in ("", "-") or text.endswith("."):
        return False
    return _number_prefix_valid(text, allow_float) and any(
        c.isdigit() for c in text
    )


def _string_token_is_valid(token_text: str) -> bool:
    """Check whether a raw token could ever be part of a JSON string.

    A token is rejected outright only if it contains an unescaped control
    character (below U+0020) before any closing double quote; everything
    else is fair game for string content.

    Args:
        token_text: Plain-text representation of the candidate token.

    Returns:
        True if the token is safe to consider as string content.
    """
    for ch in token_text:
        if ch == '"':
            return True
        if ord(ch) < 0x20:
            return False
    return True


def _string_token_consume(token_text: str) -> Tuple[bool, str]:
    """Split a token into the string content it contributes, if any.

    Args:
        token_text: Plain-text representation of the chosen token.

    Returns:
        A ``(terminated, content)`` tuple: ``terminated`` is True if the
        token contains the closing double quote, and ``content`` is the
        text to append to the value (anything before that quote, or the
        whole token if there is none).
    """
    content_chars: List[str] = []
    for ch in token_text:
        if ch == '"':
            return True, "".join(content_chars)
        content_chars.append(ch)
    return False, "".join(content_chars)


class VocabIndex:
    """Precomputed, per-vocabulary token pools used to mask logits fast.

    Scanning every token in the vocabulary at *every single generation
    step* would be wasteful. Instead we group tokens once, up front, by
    the character classes they are made of, so that at generation time we
    only need to re-check a small, relevant candidate pool.
    """

    def __init__(self, id_to_token: Dict[int, str]) -> None:
        """Build the token pools for a given vocabulary.

        Args:
            id_to_token: Mapping of token id -> raw vocabulary token.
        """
        self.text_by_id: Dict[int, str] = {
            tid: token_to_text(tok) for tid, tok in id_to_token.items()
        }

        self.number_candidates: List[int] = [
            tid
            for tid, text in self.text_by_id.items()
            if text and all(c in NUMBER_CHARS for c in text)
        ]

        self.boolean_candidates: List[int] = [
            tid
            for tid, text in self.text_by_id.items()
            if text and set(text) <= BOOLEAN_CHARS
        ]

        self.string_invalid_ids: Set[int] = {
            tid
            for tid, text in self.text_by_id.items()
            if not _string_token_is_valid(text)
        }

    def text(self, token_id: int) -> str:
        """Return the plain-text representation of *token_id*.

        Args:
            token_id: The token id to look up.

        Returns:
            The token's plain text, or "" if unknown.
        """
        return self.text_by_id.get(token_id, "")


_VOCAB_INDEX_CACHE: Dict[int, VocabIndex] = {}


def _get_vocab_index(id_to_token: Dict[int, str]) -> VocabIndex:
    """Return a cached VocabIndex for *id_to_token*, building it if needed.

    Args:
        id_to_token: Mapping of token id -> raw vocabulary token.

    Returns:
        The VocabIndex built from *id_to_token*.
    """
    key = id(id_to_token)
    cached = _VOCAB_INDEX_CACHE.get(key)
    if cached is None:
        cached = VocabIndex(id_to_token)
        _VOCAB_INDEX_CACHE[key] = cached
    return cached


def _mask_to_subset(logits: List[float], valid_ids: List[int]) -> List[float]:
    """Force every logit outside *valid_ids* to negative infinity.

    This is the literal masking step from the subject: everything that
    would break the schema gets ``-inf`` so it can never be selected.

    Args:
        logits:    Raw logits for the full vocabulary.
        valid_ids: Token ids allowed to be selected this step.

    Returns:
        A new logits list where only *valid_ids* keep their real score.
    """
    masked = [NEG_INF] * len(logits)
    for tid in valid_ids:
        if 0 <= tid < len(logits):
            masked[tid] = logits[tid]
    return masked


def _mask_excluding(logits: List[float], invalid_ids: Set[int]) -> List[float]:
    """Force every logit inside *invalid_ids* to negative infinity.

    Used when the valid set is (close to) the whole vocabulary, so it is
    cheaper to blank out the small excluded set than to rebuild it.

    Args:
        logits:      Raw logits for the full vocabulary.
        invalid_ids: Token ids that must never be selected this step.

    Returns:
        A new logits list where *invalid_ids* are set to ``-inf``.
    """
    masked = list(logits)
    for tid in invalid_ids:
        if 0 <= tid < len(masked):
            masked[tid] = NEG_INF
    return masked


def _argmax(logits: List[float]) -> Tuple[int, float]:
    """Return the (id, score) of the highest-scoring entry in *logits*.

    Args:
        logits: A list of logits, possibly containing ``-inf`` entries.

    Returns:
        Tuple of the best token id and its score.
    """
    best_id = max(range(len(logits)), key=lambda i: logits[i])
    return best_id, logits[best_id]


def _valid_number_ids(
    vocab: VocabIndex, value_text: str, allow_float: bool
) -> List[int]:
    """List every token id that could validly extend a number/integer.

    Args:
        vocab:       Precomputed vocabulary pools.
        value_text:  The value accumulated so far.
        allow_float: Whether a decimal point is permitted.

    Returns:
        Token ids that keep the value a valid numeric prefix.
    """
    return [
        tid
        for tid in vocab.number_candidates
        if _number_prefix_valid(value_text + vocab.text(tid), allow_float)
    ]


def _valid_boolean_ids(vocab: VocabIndex, value_text: str) -> List[int]:
    """List every token id that could validly extend a boolean literal.

    Args:
        vocab:      Precomputed vocabulary pools.
        value_text: The value accumulated so far.

    Returns:
        Token ids that keep the value a prefix of "true" or "false".
    """
    stripped = value_text.lstrip(" ")
    valid: List[int] = []
    for tid in vocab.boolean_candidates:
        candidate = (stripped + vocab.text(tid)).lstrip(" ")
        if candidate and any(
            lit.startswith(candidate) for lit in BOOLEAN_LITERALS
        ):
            valid.append(tid)
    return valid


def generate_value(
    prompt_ids: List[int],
    model: Any,
    vocab: VocabIndex,
    param_type: str,
    max_tokens: int = 40,
) -> Tuple[str, List[int]]:
    """Generate one parameter value using constrained, type-aware decoding.

    At every step the model's logits are masked so that only tokens
    compatible with *param_type* can ever be picked -- structurally
    invalid tokens are never even candidates, so the produced value is
    guaranteed to already be well-formed for its declared JSON type.

    Args:
        prompt_ids: The running prompt token ids (JSON written so far).
        model:      The Small_LLM_Model instance.
        vocab:      Precomputed vocabulary pools for this model.
        param_type: One of "number", "integer", "boolean", or a string
            type (anything else is treated as a JSON string).
        max_tokens: Safety cap on generated tokens.

    Returns:
        (value_text, updated_prompt_ids). ``updated_prompt_ids`` only
        includes tokens that were actually kept as part of the value (the
        caller is responsible for writing closing punctuation itself).
    """
    ids = list(prompt_ids)
    value_text = ""

    is_number = param_type in ("number", "integer")
    is_boolean = param_type == "boolean"
    allow_float = param_type == "number"

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)

        if is_number:
            valid_ids = _valid_number_ids(vocab, value_text, allow_float)
            if not valid_ids:
                break
            if _number_is_complete(value_text, allow_float):
                top_id, _ = _argmax(logits)
                if top_id not in set(valid_ids):
                    break
            masked = _mask_to_subset(logits, valid_ids)
            token_id, score = _argmax(masked)
            if score == NEG_INF:
                break
            value_text += vocab.text(token_id)
            ids.append(token_id)

        elif is_boolean:
            valid_ids = _valid_boolean_ids(vocab, value_text)
            if not valid_ids:
                break
            masked = _mask_to_subset(logits, valid_ids)
            token_id, score = _argmax(masked)
            if score == NEG_INF:
                break
            value_text += vocab.text(token_id)
            ids.append(token_id)
            if value_text.lstrip(" ") in BOOLEAN_LITERALS:
                break

        else:  # string
            masked = _mask_excluding(logits, vocab.string_invalid_ids)
            token_id, score = _argmax(masked)
            if score == NEG_INF:
                break
            terminated, content = _string_token_consume(vocab.text(token_id))
            if terminated:
                value_text += content
                break
            value_text += content
            ids.append(token_id)

    return value_text, ids


def generate_function_call(
    model: Any,
    prompt: str,
    fn: FunctionDefinition,
    id_to_token: Dict[int, str],
    encode_fn: Callable[[str], List[int]],
) -> Dict[str, Any]:
    """Fill every parameter by completing one continuous JSON object.

    A single growing prompt is built that already contains the function
    name and any parameters filled so far. Each new value is generated
    while the previous key/values are visible in the literal prompt, so
    the model stays coherent and avoids repeating values across
    parameters. We write every structural character ourselves (braces,
    keys, quotes, commas); the model only ever fills values, and those
    values are produced under constrained decoding (see ``generate_value``)
    so they are guaranteed valid for their declared type.

    Args:
        model:       The Small_LLM_Model instance.
        prompt:      The original user prompt.
        fn:          The selected function definition.
        id_to_token: Vocabulary mapping of token id to string.
        encode_fn:   Function to encode text to token ids.

    Returns:
        A dict mapping parameter name -> Python value.
    """
    vocab = _get_vocab_index(id_to_token)

    params: Dict[str, Any] = {}
    param_names = fn.get_param_names()
    param_descriptions = ", ".join(
        f"{n} ({fn.get_param_type(n)})" for n in param_names
    )

    prompt_text = (
        "/no_think\n"
        f"Function: {fn.name} - {fn.description}\n"
        f"Parameters: {param_descriptions}\n"
        f"Question: {prompt}\n"
        'Answer: {"function": "' + fn.name + '", "parameters": {'
        "Perfectly Format for Regex"

    )
    prompt_ids: List[int] = encode_fn(prompt_text)

    for pname in param_names:
        ptype = fn.get_param_type(pname)
        is_string = ptype not in ("number", "integer", "boolean")

        key_text = '"' + pname + '": '
        if is_string:
            key_text += '"'
        prompt_ids = prompt_ids + encode_fn(key_text)

        max_tokens = 64 if is_string else 24
        value_text, prompt_ids = generate_value(
            prompt_ids, model, vocab, ptype, max_tokens=max_tokens
        )

        if ptype == "number":
            params[pname] = parse_numb(value_text)
        elif ptype == "integer":
            params[pname] = int(parse_numb(value_text))
        elif ptype == "boolean":
            params[pname] = value_text.strip().lower() == "true"
        else:
            params[pname] = value_text

        closer = '", ' if is_string else ", "
        prompt_ids = prompt_ids + encode_fn(closer)

    return params


def _sequence_logprob(
    model: Any,
    context_ids: List[int],
    prompt_ids: List[int],
) -> float:
    """Compute the average per-token log-probability of a continuation.

    Uses teacher forcing: at each step the true previous tokens are fed
    back in, and we read how much probability the model assigned to the
    actual next token.

    Args:
        model:       The Small_LLM_Model instance.
        context_ids: Token ids of the shared prompt/context.
        prompt_ids:  Token ids of the text to score as a continuation.

    Returns:
        Average log-probability per token, in (-inf, 0].
    """
    ids = list(context_ids)
    total = 0.0
    for tok_id in prompt_ids:
        logits = model.get_logits_from_input_ids(ids)
        max_logit = max(logits)
        logsumexp = max_logit + math.log(
            sum(math.exp(logit - max_logit) for logit in logits)
        )
        total += logits[tok_id] - logsumexp
        ids.append(tok_id)
    return total / max(len(prompt_ids), 1)


def select_function(
    model: Any,
    prompt: str,
    functions: List[FunctionDefinition],
    encode_fn: Callable[[str], List[int]],
) -> FunctionDefinition:
    """Select the function under which the user's request is most probable.

    For each candidate function, build a short context describing it, then
    measure how likely the model finds the actual user prompt as a
    continuation of that context. The function whose context makes the
    prompt most probable wins. Because the scored continuation (the
    prompt) is identical across candidates, there is no length bias, and
    the choice is made purely from the LLM's own probabilities -- no
    keyword heuristics involved.

    Args:
        model:     The Small_LLM_Model instance.
        prompt:    The natural language user request.
        functions: List of available FunctionDefinition objects.
        encode_fn: Function to encode text to token ids.

    Returns:
        The FunctionDefinition under which the prompt is most probable.

    Raises:
        ValueError: If *functions* is empty.
    """
    if not functions:
        raise ValueError("No functions available to select from.")
    if not prompt.strip():
        un = "unknown"
        prompt_ids = encode_fn(un)
    else:
        prompt_ids = encode_fn(prompt)

    best_fn = functions[0]
    best_score = NEG_INF
    for fn in functions:
        context = f"Function: {fn.name} - {fn.description}\nRequest: "
        context_ids = encode_fn(context)
        score = _sequence_logprob(model, context_ids, prompt_ids)
        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn