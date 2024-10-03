import torch
import stdlib.jsonx as jsonx
from diffusers import StableDiffusion3Pipeline

HF_TOKEN = jsonx.load("../tokens.json")['hf']

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    token=HF_TOKEN)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

image.save("images/stabilityai-3.png")
