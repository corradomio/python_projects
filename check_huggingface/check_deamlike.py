#
# https://dreamlike.art/
#

from diffusers import StableDiffusionPipeline
import torch

# model_id = "dreamlike-art/kandinsky-2.1"        #DOESN'T work
# model_id = "dreamlike-art/dreamlike-diffusion-1.0"
# model_id = "dreamlike-art/dreamlike-photoreal-1.0"
# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
model_id = "dreamlike-art/dreamlike-anime-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "caucasian creative man wearing a sweater, sitting, on an icelandic beach"
image = pipe(prompt).images[0]

image.save("./images/dreamlike-caucasian_man.jpg")
