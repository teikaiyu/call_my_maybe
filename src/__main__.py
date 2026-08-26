# import sys
# # import pydantic
# # from src.llm_sdk.llm_sdk import Small_LLM_Model


# # def main():
# #     llm = Small_LLM_Model()
# #     ids = llm.encode("Hello")
# #     print(ids)
# #     # print(len(llm.get_logits_from_input_ids(ids[0].tolist())))
# #     print(llm.decode(ids))
# #     print()


# def call_me_maybe(funcs_def: str, inputf: str, outputf: str) -> None:
#     parse_funcs_def(funcs_def, inputf)
#     parse_prompt(inputf)
#     llm_generate(outputf)


# if __name__ == "__main__":
#     args: list[str] = sys.argv[1:]
#     funcs_def: str = "data/input/functions_definition.json"
#     inputf: str = "data/input/functionn_calling_tests.json"
#     outputf: str = "data/output/output.json"

#     i = 0
#     while i + 1 < len(args):
#         v = args[i]
#         if v == "--functions_definition":
#             funcs_def = args[i + 1]
#         elif v == "--input":
#             inputf = args[i + 1]
#         elif v == "--output":
#             outputf = args[i + 1]
#         else:
#             raise ValueError(f"Invalid option: {v}")
#         i += 2

#     call_me_maybe(funcs_def, inputf, outputf)


from .llm_sdk.llm_sdk import Small_LLM_Model
import json


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
