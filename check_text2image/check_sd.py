import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)


pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello Fast World",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
image.save("capybara-1.png")
