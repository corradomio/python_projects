#
# https://blog.salesforceairesearch.com/codet5-open-code-large-language-models/
#
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m"
# checkpoint = "Salesforce/codet5p-220m-py"
# checkpoint = "Salesforce/codet5p-770m"
# checkpoint = "Salesforce/codet5p-770m=py"
# checkpoint = "Salesforce/codet5p-2b"
# checkpoint = "Salesforce/codet5p-6b"
# checkpoint = "Salesforce/codet5p-16b"


device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
