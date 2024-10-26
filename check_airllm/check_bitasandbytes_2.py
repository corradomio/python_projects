from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig

# model_id="bigscience/bloomz-560m"
# model_id="bigscience/bloomz-1b1"
# model_id="bigscience/bloomz-1b7"
# model_id="bigscience/bloomz-3b"
# model_id="bigscience/bloomz-7b1"
model_id="bigscience/bloomz"


qconfig = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=qconfig
)

tokenizer = AutoTokenizer.from_pretrained(model_id)


MAX_LENGTH = 128
input_text = [
    'What is the capital of United States?',
    #'I like',
]

input_tokens = tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               padding=False)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = tokenizer.decode(generation_output.sequences[0])

print(output)
