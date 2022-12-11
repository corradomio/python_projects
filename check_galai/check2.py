from transformers import AutoTokenizer, OPTForCausalLM

input_text = "The Transformer architecture [START_REF]"

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
r = tokenizer.decode(outputs[0])
print(r)
