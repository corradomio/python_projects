import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)


pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float32
).to('cuda')

prompt = (
    "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, "
    "jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
)
with autocast("cuda"):
    res =  pipe(prompt, guidance_scale=6)
    print(res)
    image = res["images"][0]

image.save("waifu.png")
