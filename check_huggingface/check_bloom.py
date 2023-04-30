import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch
print(torch.cuda.is_available())

HF_TOKEN = "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"

# model_id = "bigscience/bloom-1b7"
# model_id = "bigscience/bloom-560m"
model_id = "bigscience/bloom-3b"

print("Load model")
tokenizer = BloomTokenizerFast.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = BloomForCausalLM.from_pretrained(model_id, use_auth_token=HF_TOKEN)

print("To cuda")
model.to('cuda:0')

prompt = "It was a dark and stormy night"
print(prompt)
result_length = 100
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')

print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length
                      )[0]))

print("\n--\n")
prompt = "italiano: era una notte buia e tempestosa"
print(prompt)
result_length = 100
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')

print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length
                      )[0]))

