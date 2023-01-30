from transformers import pipeline, set_seed
from pprint import pprint

generator = pipeline('text-generation', model='gpt2')

set_seed(42)

res = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

pprint(res)


from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
pprint(output)


