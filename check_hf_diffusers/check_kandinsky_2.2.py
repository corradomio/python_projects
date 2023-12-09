from diffusers import AutoPipelineForImage2Image
import torch
import requests
from io import BytesIO
from PIL import Image
import os

pipe = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()

prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
original_image = Image.open(BytesIO(response.content)).convert("RGB")
original_image.thumbnail((768, 768))

image = pipe(prompt=prompt, image=original_image, strength=0.3).images[0]
image.save("images/kandinsky-fantasy_land.png")