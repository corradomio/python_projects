#
# https://huggingface.co/docs/diffusers/quicktour?diffusionpipeline=text-to-image
#
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)


quant_config = PipelineQuantizationConfig(
  quant_backend="bitsandbytes_4bit",
  quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
  components_to_quantize=["transformer", "text_encoder"],
)
pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image",
  torch_dtype=torch.bfloat16,
  quantization_config=quant_config,
  device_map="cuda"
)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


image.save("diff3.png")


