from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("v2ray/Llama-3-70B")
input_text = [
    'What is the capital of United States?'
]

print(type(model))

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
