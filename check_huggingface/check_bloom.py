import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch


HF_TOKEN = "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"

model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3", use_auth_token=HF_TOKEN)
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3", use_auth_token=HF_TOKEN)

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length
                      )[0]))

