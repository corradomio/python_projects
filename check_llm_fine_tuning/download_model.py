import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)
repo_id = 'microsoft/Phi-3-mini-4k-instruct'

model = AutoModelForCausalLM.from_pretrained(
   repo_id, device_map="cuda:0", quantization_config=bnb_config
)