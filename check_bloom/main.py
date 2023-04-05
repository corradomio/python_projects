from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

from transformers import pipeline

text2text_generator = pipeline("text2text-generation", model="bigscience/bloomz-560m")
text = text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
print(text)
# [{'generated_text': 'the answer to life, the universe and everything'}]

text = text2text_generator("translate from English to French: I'm very happy")
print(text)
# [{'generated_text': 'Je suis très heureux'}]


from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
text = generator("Hello, I'm a language model", max_length=30, num_return_sequences=3)
print(text)


generator = pipeline('text-generation', model='bigscience/bloomz-560m')
text = generator("Hello, I'm a language model", max_length=30, num_return_sequences=1)
print(text)
