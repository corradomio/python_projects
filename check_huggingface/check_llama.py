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

from transformers import pipeline, set_seed, LlamaTokenizer
from pprint import pprint

model = 'decapoda-research/llama-7b-hf'
# model = 'decapoda-research/llama-13b-hf'

generator = pipeline('text-generation', model=model)

set_seed(42)

text = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""

# text = "The Artificial Intelligence,"

res = generator(text, max_length=200, num_return_sequences=1)

pprint(res)
