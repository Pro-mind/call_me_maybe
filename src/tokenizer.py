"""Custom BPE tokenizer using vocab.json and merges.txt files.

Implements byte-level BPE compatible with Qwen/GPT-2 style vocabularies,
without relying on the model's built-in encode/decode methods.

Usage::

    tokenizer = BPETokenizer(vocab_path, merges_path)
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)
"""

import json
import re
from typing import Dict, List, Optional, Tuple


def _build_byte_to_unicode() -> Dict[int, str]:
    """Build the GPT-2 style byte→unicode mapping.

    Every possible byte (0-255) maps to a unique unicode character.
    Printable ASCII-ish bytes map to themselves; the rest get mapped
    to codepoints starting at 256 so they are unambiguous.

    Returns:
        Dict mapping byte value (int) to unicode character (str).
    """
    bs: List[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENC: Dict[int, str] = _build_byte_to_unicode()
_BYTE_DEC: Dict[str, int] = {v: k for k, v in _BYTE_ENC.items()}

# Simplified GPT-2 style pre-tokenizer pattern (works for ASCII text)
_PRE_TOK_PAT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"   # English contractions
    r"| ?\d+"                    # optional-space + digits
    r"| ?[a-zA-Z]+"              # optional-space + letters
    r"| ?[^ \t\n\r\f\va-zA-Z0-9]+"  # optional-space + other non-whitespace
    r"|\s+"                      # pure whitespace
)


class BPETokenizer:
    """Byte-level BPE tokenizer compatible with Qwen/GPT-2 vocabularies.

    Implements encode and decode without using the model's built-in methods.
    Uses only the vocabulary file and merges file exposed by the SDK.

    Args:
        vocab_path:   Path to vocab.json  (token_str -> token_id).
        merges_path:  Path to merges.txt  (BPE merge rules, one per line).
    """

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as fv:
            raw_vocab: Dict[str, int] = json.load(fv)

        self._tok2id: Dict[str, int] = raw_vocab
        self._id2tok: Dict[int, str] = {v: k for k, v in raw_vocab.items()}

        # Parse merges: rank = line order (lower = higher priority)
        self._merges: Dict[Tuple[str, str], int] = {}
        with open(merges_path, "r", encoding="utf-8") as fm:
            rank = 0
            for line in fm:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    self._merges[(parts[0], parts[1])] = rank
                    rank += 1

        self._bpe_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Read-only id→token-string mapping (same format as load_vocab)."""
        return self._id2tok

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of token IDs using byte-level BPE.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        ids: List[int] = []
        for chunk in _PRE_TOK_PAT.findall(text):
            # Map each UTF-8 byte to its unicode representative character
            byte_str = "".join(_BYTE_ENC[b] for b in chunk.encode("utf-8"))
            for piece in self._apply_bpe(byte_str):
                token_id = self._tok2id.get(piece)
                if token_id is not None:
                    ids.append(token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        byte_chars = "".join(self._id2tok.get(i, "") for i in ids)
        raw_bytes = bytearray(
            _BYTE_DEC[c] for c in byte_chars if c in _BYTE_DEC
        )
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # BPE algorithm
    # ------------------------------------------------------------------

    def _apply_bpe(self, token: str) -> List[str]:
        """Apply BPE merges to a single byte-level unicode token.

        Greedily merges the highest-priority (lowest-rank) pair at each step,
        repeating until no more merges are applicable.

        Args:
            token: Byte-level unicode string after pre-tokenization.

        Returns:
            List of sub-token strings after all applicable merges.
        """
        if token in self._bpe_cache:
            return self._bpe_cache[token]

        word: List[str] = list(token)

        while len(word) > 1:
            best_rank: Optional[int] = None
            best_pair: Optional[Tuple[str, str]] = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # no more applicable merges

            a, b = best_pair
            merged = a + b
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        self._bpe_cache[token] = word
        return word
