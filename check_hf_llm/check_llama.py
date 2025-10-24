from transformers import LlamaForCausalLM, LlamaTokenizer

tokdir = "E:/Datasets/NLP/LLAMA.huggingface"
modeldir = "E:/Datasets/NLP/LLAMA.huggingface/7B"

tokenizer = LlamaTokenizer.from_pretrained(tokdir)
model = LlamaForCausalLM.from_pretrained(modeldir)


text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=20)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "user: {user.name}"
