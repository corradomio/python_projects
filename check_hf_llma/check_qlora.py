#
# https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing
#
# Bitsandbytes for windows latest version
# https://github.com/jllllll/bitsandbytes-windows-webui
#
#
# python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.39.0-py3-none-any.whl
# pip install --upgrade accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(model)

# --

text = "Hello my name is"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# --
