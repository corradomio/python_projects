from diffusers import StableDiffusionPipeline
import torch

#
# STRA Gnocche
# https://huggingface.co/DucHaiten/DucHaiten-StyleLikeMe
#

model_id = "DucHaiten/DucHaitenAIart"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/duchaiten-1.jpg")


model_id = "DucHaiten/DucHaiten-StyleLikeMe"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/duchaiten-2.jpg")
