from transformers import BloomTokenizerFast

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")

print(tokenizer("Hello world")['input_ids'])
print(tokenizer(" Hello world")['input_ids'])
