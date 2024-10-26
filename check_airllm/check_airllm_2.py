from airllm import AutoModel
import stdlib.jsonx as jsonx


HF_TOKEN = jsonx.load("../key_tokens.json")['hf']


# model_id = "bigscience/bloom-1b7"
# model_id = "bigscience/bloom-560m"
model_id = "bigscience/bloom-3b"

print("Load model")

model = AutoModel.from_pretrained(model_id, compression='4bit')


MAX_LENGTH = 128
input_text = [
    'What is the capital of United States?',
    #'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               padding=False)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
