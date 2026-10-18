import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer

model = "openai-community/gpt2"

tokenizer:GPT2Tokenizer = AutoTokenizer.from_pretrained(model)
print(tokenizer.__class__)
vocab_size = len(tokenizer.get_vocab())
print("vocab size:", vocab_size)
model = AutoModel.from_pretrained(model)
print(model.__class__)


sentence = "The Transformer has revolutionized NLP."
print("original sentence:", sentence)

# inputs = tokenizer(sentence, return_tensors="pt")
# input_ids = inputs["input_ids"]
# print(input_ids)

tokens = tokenizer.tokenize(sentence)
print("tokens:", tokens)

input_ids = tokenizer.encode(sentence)
print("input ids:", input_ids)

sentence = tokenizer.decode(input_ids)
print("recreated sentence:", sentence)

sentence = tokenizer.decode(list(range(20)))
print("special sentence/1:", sentence)

sentence = tokenizer.decode(list(range(vocab_size-10, vocab_size+1)))
print("special sentence/2:", sentence)


input_ids = torch.tensor(input_ids)
outputs = model(input_ids)
embeddings = outputs.last_hidden_state
print("embeddings:", embeddings)


