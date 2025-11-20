#
# https://huggingface.co/docs/diffusers/quicktour?diffusionpipeline=text-to-image
#
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)


pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)

pipeline.enable_model_cpu_offload()

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline(prompt).images[0]


image.save("diff1.png")
