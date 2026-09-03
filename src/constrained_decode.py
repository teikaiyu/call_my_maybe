"""Constrained decoding: force the model to emit valid function-calling JSON.

Core idea: at every generation step, restrict the vocab to tokens that keep
the output consistent with the JSON structure and the schema
(function name enum, parameter names/types from functions_definition.json).

Known/fixed parts (braces, key names, the prompt value itself) are forced
literally. Only two things are genuinely uncertain and need real constraint:
    - "name": must match one of the function names (enum)
    - each parameter value: constrained by its declared type

This is a first working version, not exhaustive edge-case handling
(string escaping, unicode edge cases, etc. are simplified).
"""
import json
from typing import Any, Iterable
from llm_sdk import Small_LLM_Model


# --- GPT style byte-level BPE decoder (standard, public algorithm) -----------

def _bytes_to_unicode() -> dict[int, str]:
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("i"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    bs_copy = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            bs_copy.append(256 + n)
            n += 1
    cs: list[str] = [chr(n) for n in bs_copy]
    return dict(zip(bs, cs))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def _token_to_string(token: str) -> str:
    """Convert a raw vocab entry (e.g. "Ġhello") into its real text."""
    try:
        return bytes(_BYTE_DECODER[c] for c in token if c in _BYTE_DECODER
                     ).decode("utf-8", errors="ignore")
    except (KeyError, UnicodeDecodeError):
        return token  # special tokens etc. fall back to raw form


def load_vocab(vocab_path: str) -> dict[int, str]:
    with open(vocab_path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)
    return {tok_id: _token_to_string(tok) for tok, tok_id in raw.items()}


# --- constrained generation engine--------------------------------------------

class ConstrainedJSONGenerator:
    def __init__(
            self,
            model: Small_LLM_Model,
            vocab: dict[int, str],
            functions: list[dict[str, Any]],
            prompt: str
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.functions = {f["name"]: f for f in functions}
        self.prompt = prompt

        self.vocab_trie = self._build_vocab_trie()
        self.enum_trie = self._build_trie(self.functions.keys())
        self.bool_trie = self._build_trie(["true", "false"])

        self.number_tok_ids = {
            tid for tid, s in self.vocab.items()
            if len(s) > 0 and all(c in "0123456789.-" for c in s)
        }
        self.free_string_tok_ids = {
            tid for tid, s in self.vocab.items()
            if len(s) > 0 and '"' not in s and '\\' not in s
        }

        self._quote_tok_id = self._find_tok_id('"')
        self._comma_tok_id = self._find_tok_id(',')
        self._close_brace_tok_id = self._find_tok_id('}')

    def _find_tok_id(self, c: str) -> int:
        return next(tid for tid, s in self.vocab.items() if s == c)

    def _build_trie(self, vocab: Iterable[str]) -> dict[str | None, Any]:
        trie: dict[str | None, Any] = {}
        for word in vocab:
            current = trie
            for char in word:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current[None] = True
        return trie

    def _build_vocab_trie(self) -> dict[str | None, Any]:
        trie: dict[str | None, Any] = {}
        for tok_id, tok_str in self.vocab.items():
            if len(tok_str) == 0:
                continue
            current = trie
            for char in tok_str:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current[None] = tok_id
        return trie

    def _walk(
            self,
            enum_node: dict[str | None, Any],
            vocab_node: dict[str | None, Any],
            candids: dict[int, Any]
    ) -> None:
        """Simultaneously descend enum_trie and vocab_trie. Whenever vocab_node
        marks the end of a real token, record that token_id as a valid choice,
        mapped to the enum_trie position it would leave us at. Mutates
        `candids` in place; returns nothing."""
        if None in vocab_node:
            candids[vocab_node[None]] = enum_node
        for char in vocab_node:
            if char is None:
                continue
            if char in enum_node:
                self._walk(enum_node[char], vocab_node[char], candids)

    def _mask_and_pick(
            self, input_ids: list[int], allowed_ids: set[int]
    ) -> int:
        """Get logits, zero out (via -inf) everything invalid, argmax the rest.
        """
        logits = self.model.get_logits_from_input_ids(input_ids)
        best_id, best_val = None, float("-inf")
        for tok_id in allowed_ids:
            if tok_id < len(logits) and logits[tok_id] > best_val:
                best_id, best_val = tok_id, logits[tok_id]
        if best_id is None:
            raise RuntimeError("no valid token for current constraint state")
        return best_id

    def _force_literal(self, input_ids: list[int], literal: str) -> list[int]:
        # remaining = literal
        # while remaining:
        #     allowed: dict[int, Any] = {}
        #     node = self.vocab_trie
        #     for c in remaining:
        #         if c not in node:
        #             break
        #         node = node[c]
        #         if None in node:
        #             allowed[node[None]] = None
        #     tok_id = self._mask_and_pick(input_ids, set(allowed.keys()))
        #     tok_str = self.vocab[tok_id]
        #     input_ids.append(tok_id)
        #     remaining = remaining[len(tok_str):]
        # return input_ids
        literal_ids: list[int] = self.model.encode(literal)[0].tolist()
        return input_ids + literal_ids

    def _generate_enum(self, input_ids: list[int]) -> tuple[list[int], str]:
        """Generate a quoted string value that must exactly match one of
        `options`."""
        enum_node = self.enum_trie
        collected = ""
        while True:
            candids: dict[int, Any] = {}
            self._walk(enum_node, self.vocab_trie, candids)
            if None in enum_node:
                candids[self._quote_tok_id] = enum_node
            tok_id = self._mask_and_pick(input_ids, set(candids.keys()))
            tok_str = self.vocab[tok_id]
            input_ids.append(tok_id)

            if tok_str == '"' and None in enum_node:
                break

            collected += tok_str
            enum_node = candids[tok_id]
        return input_ids, collected

    def _generate_enum_bare(
            self, input_ids: list[int]
    ) -> tuple[list[int], str]:
        """Generate an unquoted true/false literal"""
        bool_node = self.bool_trie
        collected = ""

        while True:
            candids: dict[int, Any] = {}
            self._walk(bool_node, self.vocab_trie, candids)

            if None in bool_node:
                candids[self._comma_tok_id] = bool_node
                candids[self._close_brace_tok_id] = bool_node

            tok_id = self._mask_and_pick(input_ids, set(candids.keys()))
            tok_str = self.vocab[tok_id]

            if tok_str in (",", "}") and None in bool_node:
                break

            input_ids.append(tok_id)
            collected += tok_str
            bool_node = candids[tok_id]

        return input_ids, collected

    def _generate_number(
            self, input_ids: list[int]
    ) -> tuple[list[int], str]:
        """Digits/./- only. Terminators (',' or '}') are included as valid
        choices so the model can naturally signal 'done' without us guessing
        a fixed length; they are NOT consumed here, _force_literal adds them.
        """
        collected = ""
        while True:
            allowed = set(self.number_tok_ids)
            if collected:
                allowed.add(self._comma_tok_id)
                allowed.add(self._close_brace_tok_id)
            tok_id = self._mask_and_pick(input_ids, allowed)
            tok_str = self.vocab[tok_id]
            if tok_str in (",", "}"):
                break
            input_ids.append(tok_id)
            collected += tok_str
        return input_ids, collected

    def _generate_free_string(
            self, input_ids: list[int], max_chars: int = 200
    ) -> tuple[list[int], str]:
        """Any characters except unexcaped quote/backslash, untillslogin quote.
        """
        collected = ""
        while len(collected) < max_chars:
            allowed = set(self.free_string_tok_ids)
            allowed.add(self._quote_tok_id)

            tok_id = self._mask_and_pick(input_ids, allowed)
            tok_str = self.vocab[tok_id]
            input_ids.append(tok_id)
            if tok_str == '"':
                break
            collected += tok_str
        return input_ids, collected

    def generate(self) -> dict[str, Any]:
        functions_desc = "\n".join(
            f'- {fn["name"]}: {fn["description"]}'
            for fn in self.functions.values()
        )
        system_context = f"Available functions:\n{functions_desc}\n\nQuery: "
        input_ids = self.model.encode(system_context + self.prompt)[0].tolist()

        input_ids = self._force_literal(input_ids, '{"prompt": "')
        input_ids = self._force_literal(input_ids, self.prompt)
        input_ids = self._force_literal(input_ids, '", "name": "')
        input_ids, fn_name = self._generate_enum(input_ids)
        input_ids = self._force_literal(input_ids, '", "parameters": {')

        fn_def = self.functions[fn_name]
        params: dict[str, Any] = fn_def["parameters"]
        collected_params: dict[str, Any] = {}
        for i, (param_name, param_schema) in enumerate(params.items()):
            input_ids = self._force_literal(input_ids, f'"{param_name}": ')
            value_type = param_schema["type"]

            if value_type == "string":
                input_ids = self._force_literal(input_ids, '"')
                input_ids, value = self._generate_free_string(input_ids)
                collected_params[param_name] = value
            elif value_type == "number":
                input_ids, value = self._generate_number(input_ids)
                collected_params[param_name] = float(value)
            elif value_type == "boolean":
                input_ids, value = self._generate_enum_bare(input_ids)
                collected_params[param_name] = value == "true"
            else:
                raise ValueError(f"unsupported parameter type: {value_type}")

            if i < len(params) - 1:
                input_ids = self._force_literal(input_ids, ", ")

        input_ids = self._force_literal(input_ids, "}}")

        return {"prompt": self.prompt,
                "name": fn_name,
                "parameters": collected_params}
