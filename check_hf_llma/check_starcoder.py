#
# https://huggingface.co/bigcode/starcoder
#
# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token='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')

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
