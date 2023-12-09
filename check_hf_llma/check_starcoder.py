#
# https://huggingface.co/bigcode/starcoder
#
# pip install -q transformers
import jsonx
HF_TOKEN = jsonx.load("../tokens.json")['hf']


from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token=HF_TOKEN)

checkpoint = "bigcode/starcoder"
device = "cuda"     # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
