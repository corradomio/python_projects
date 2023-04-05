# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# checkpoint = "decapoda-research/llama-7b-hf"
#
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
#
# inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))

from transformers import pipeline, set_seed
from pprint import pprint

model = 'decapoda-research/llama-7b-hf'
# model = 'decapoda-research/llama-13b-hf'

generator = pipeline('text-generation', model=model)

set_seed(42)

res = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)

pprint(res)
