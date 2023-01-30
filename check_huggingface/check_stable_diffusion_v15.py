import torch

model_id = "runwayml/stable-diffusion-v1-5"

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(model_id)

prompt = "portrait photo of a old warrior chief"

pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipe(prompt, generator=generator).images[0]
image.save("portrait_photo_of_a_old_warrior_chief.png")
