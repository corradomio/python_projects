import os
import torch
import transformers
from diffusers import StableDiffusionPipeline

os.environ["PYTORCH_CUDA_ALLOC_CON"] = "max_split_size_mb:14336"

prompt = "beautiful young girl, high resolution"

# model = "prompthero/openjourney"
#
# pipeline = StableDiffusionPipeline.from_pretrained(model)
# pipeline.to("cuda")
#
# image = pipeline(prompt).images[0]
# image.save("images/prompthero-1.png")

model = "prompthero/openjourney-v2"

pipeline = StableDiffusionPipeline.from_pretrained(model)
pipeline.to("cuda")

image = pipeline(prompt).images[0]
image.save("images/prompthero-2.png")

model = "prompthero/openjourney-v4"

pipeline = StableDiffusionPipeline.from_pretrained(model)
pipeline.to("cuda")

image = pipeline(prompt).images[0]
image.save("images/prompthero-4.png")
