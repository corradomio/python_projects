# --

import torch
from transformers import pipeline

generate_text = pipeline(model="h2oai/h2ogpt-oig-oasst1-512-6.9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
print(res[0]["generated_text"])


exit()
# --

# import torch
# from transformers import pipeline
#
# generate_text = pipeline(model="h2oai/h2ogpt-oasst1-512-20b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
#
# res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
# print(res[0]["generated_text"])

# --

import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-512-20b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-512-20b", torch_dtype=torch.bfloat16, device_map="auto")
generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
print(res[0]["generated_text"])
