import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
access_token="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"

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=access_token)
pipe = pipe.to(device)

# prompt = "a photo of an astronaut riding a horse on mars"
# for i in range(10):
#     with autocast("cuda"):
#         image = pipe(prompt, guidance_scale=7.5)["sample"][0]
#     image.save(f"astronaut_rides_horse-{i}.png")

# prompt = 'Chinese architecture'
# for i in range(10):
#     with autocast("cuda"):
#         image = pipe(prompt, guidance_scale=7.5)["sample"][0]
#     image.save(f"chinese_architecture-{i}.png")

# prompt = 'tokyo house'
# for i in range(10):
#     with autocast("cuda"):
#         image = pipe(prompt, guidance_scale=7.5)["sample"][0]
#     image.save(f"tokyo-house-{i}.png")

# prompt = "A dream of a distant galaxy, by Caspar David Friedrich, matte painting trending on artstation HQ"
# for i in range(10):
#     with autocast("cuda"):
#         res = pipe(prompt, guidance_scale=7.5)
#         image = res["sample"][0]
#     image.save(f"distant-galaxy-{i}.png")

prompt = "robot dancing in the rain with fire made from cotton candy, hyper realistic, photorealism, 4k, sophisticated, octane render"
for i in range(10):
    with autocast("cuda"):
        res = pipe(prompt, guidance_scale=7.5)
        image = res["sample"][0]
    image.save(f"robot-dancing-{i}.png")
