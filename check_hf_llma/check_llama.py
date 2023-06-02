from transformers import LlamaForCausalLM, LlamaTokenizer

tokdir = "E:/Datasets/NLP/LLAMA.huggingface"
modeldir = "E:/Datasets/NLP/LLAMA.huggingface/7B"

tokenizer = LlamaTokenizer.from_pretrained(tokdir)
model = LlamaForCausalLM.from_pretrained(modeldir)
