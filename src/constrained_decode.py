import json
from typing import Any, Callable
from llm_sdk import Small_LLM_Model


# --- GPT style byte-level BPE decoder (standard, public algorithm) ---

def _bytes_to_unicode() -> dict[int, str]:
    """256種類のバイト値と256種類の'印字可能な'文字の変換表。文字←→バイト値の可逆性確保
    BPEトークナイザのバイト単位処理をしやすくするために、画面に表示できない、しづらい制御文字や空白などを印字できる普通の文字に付け替えるため

    bs: 印字可能だから触らなくてもよいバイト値たち  # ¬:u00ac, ®:u00ae, ÿ:u00ff
    """
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
    try:
        return bytes(_BYTE_DECODER[c] for c in token if c in _BYTE_DECODER
                     ).decode("utf-8", errors="ignore")
    except (KeyError, UnicodeDecodeError):
        return token


def load_vocab(vocab_path: str) -> dict[int, str]:
    with open(vocab_path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)
    return {tok_id: _token_to_string(tok) for tok, tok_id in raw.items()}


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

    def _mask_and_pick(
            self, input_ids: list[int], is_valid: Callable[[str], bool]
    ) -> int:
        """Get logits, zero out (via -inf) everything invalid, argmax the rest.
        """
        logits = self.model.get_logits_from_input_ids(input_ids)
        best_id, best_val = None, float("-inf")
        for tok_id, tok_str in self.vocab.items():
            if tok_id >= len(logits):
                continue
            if is_valid(tok_str) and logits[tok_id] > best_val:
                best_id, best_val = tok_id, logits[tok_id]
        if best_id is None:
            raise RuntimeError("no valid token for current constraint state")
        return best_id

    def _force_literal(self, input_ids: list[int], literal: str) -> list[int]:
        literal_ids = self.model.encode(literal)[0].tolist()
        return input_ids + literal_ids

    def _generate_enum(
            self, input_ids: list[int], options: list[str]
    ) -> tuple[list[int], str]:
        """Generate a quoted string value that must exactly match one of
        `options`."""
        remaining_options = options
        collected = ""
        while True:
            def is_valid(tok_str: str) -> bool:
                if collected in options and tok_str == '"':
                    return True
                return len(tok_str) > 0 and any(
                    opt[len(collected):].startswith(tok_str)
                    for opt in remaining_options
                )

            tok_id = self._mask_and_pick(input_ids, is_valid)
            tok_str = self.vocab[tok_id]
            input_ids.append(tok_id)
            if tok_str == '"' and collected in options:
                break
            collected += tok_str
            remaining_options = [o for o in remaining_options
                                 if o.startswith(collected)]
        return input_ids, collected

    def _generate_enum_bare(
            self, input_ids: list[int], options: list[str]
    ) -> tuple[list[int], str]:
        """Same as _generate_enum but for unquoted literals (true/false)."""
        remaining_options = options
        collected = ""
        while collected not in options:
            def is_valid(tok_str: str) -> bool:
                return len(tok_str) > 0 and any(
                    opt[len(collected):].startswith(tok_str)
                    for opt in remaining_options
                )

            tok_id = self._mask_and_pick(input_ids, is_valid)
            tok_str = self.vocab[tok_id]
            input_ids.append(tok_id)
            collected += tok_str
            remaining_options = [o for o in remaining_options
                                 if o.startswith(collected)]
        return input_ids, collected

    def _generate_number(
            self, input_ids: list[int]
    ) -> tuple[list[int], str]:
        """Digits/./- only. Terminators (',' or '}') are included as valid
        choices so the model can naturally signal 'done' without us guessing
        a fixed length; they are NOT consumed here, _force_literal adds them.
        """
        allowed = set("0123456789.-")
        collected = ""
        while True:
            def is_valid(tok_str: str) -> bool:
                if tok_str in (",", "}") and collected:
                    return True
                return len(tok_str) > 0 and all(c in allowed for c in tok_str)

            tok_id = self._mask_and_pick(input_ids, is_valid)
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
            def is_valid(tok_str: str) -> bool:
                return tok_str == '"' or ('"' not in tok_str
                                          and "\\" not in tok_str)

            tok_id = self._mask_and_pick(input_ids, is_valid)
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
        input_ids, fn_name = self._generate_enum(input_ids,
                                                 list(self.functions.keys()))
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
                input_ids, value = self._generate_enum_bare(input_ids,
                                                            ["true", "false"])
                collected_params[param_name] = value == "true"
            else:
                raise ValueError(f"unsupported parameter type: {value_type}")

            if i < len(params) - 1:
                input_ids = self._force_literal(input_ids, ", ")

        input_ids = self._force_literal(input_ids, "}}")

        return {"prompt": self.prompt,
                "name": fn_name,
                "parameters": collected_params}


if __name__ == "__main__":
    model = Small_LLM_Model()
    vocab = load_vocab(model.get_path_to_vocab_file())

    with open("data/input/functions_definition.json", encoding="utf-8") as f:
        functions = json.load(f)
    with open("data/input/function_calling_tests.json", encoding="utf-8") as f:
        test_prompts = json.load(f)

    results = []
    # for prompt in test_prompts:
    #     prompt = prompt["prompt"]
    #     gen = ConstrainedJSONGenerator(model, vocab, functions, prompt)
    #     results.append(gen.generate())
    prompt = "What is the sum of 2 and 3?"
    gen = ConstrainedJSONGenerator(model, vocab, functions, prompt)
    results.append(gen.generate())
    print(json.dumps(results, indent=2))
