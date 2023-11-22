import os
import torch
import transformers
from diffusers import DiffusionPipeline

os.environ["PYTORCH_CUDA_ALLOC_CON"] = "max_split_size_mb:14336"

# --

# model = "stabilityai/stable-diffusion-2-1"
# model = "stabilityai/stable-diffusion-2-1-base"
# model = "stabilityai/stable-diffusion-2-1-unclip"
# model = "stabilityai/stable-diffusion-2"
model = "stabilityai/stable-diffusion-xl-base-1.0"

pipeline = DiffusionPipeline.from_pretrained(model)
pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("images/stabilityai-1.png")

# --

# from diffusers import StableDiffusionInpaintPipeline
#
# model = "stabilityai/stable-diffusion-2-inpainting"
#
# pipeline = StableDiffusionInpaintPipeline.from_pretrained(model)
# pipeline.to("cuda")
#
# image = pipeline(prompt).images[0]
# image.save("images/stabilityai-2.png")
