from diffusers import StableDiffusionPipeline
import torch

# model_id = "prompthero/openjourney"
# model_id = "prompthero/openjourney-v2"
model_id = "prompthero/openjourney-v4"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
image = pipe(prompt).images[0]
image.save("./images/retro_cars_4.png")
