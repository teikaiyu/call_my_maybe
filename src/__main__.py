from .llm_sdk.llm_sdk import Small_LLM_Model


def generate(model: Small_LLM_Model, prompt: str) -> str:
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    eos_id = model._tokenizer.eos_token_id
    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids)
        next_ids = max(range(len(logits)), key=lambda i: logits[i])
        input_ids.append(next_ids)

        if next_ids == eos_id:
            break

    return model.decode(input_ids)


if __name__ == "__main__":
    model = Small_LLM_Model()
    result = generate(model, "Hello world")
    print(result)
