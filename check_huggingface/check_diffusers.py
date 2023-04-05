import os
import torch
import transformers
from diffusers import DiffusionPipeline

os.environ["PYTORCH_CUDA_ALLOC_CON"] = "max_split_size_mb:8192"

# brutte immagini
# model = "runwayml/stable-diffusion-v1-5"

model = "stabilityai/stable-diffusion-2-1"
# model = "stabilityai/stable-diffusion-2-1-base"
# model = "stabilityai/stable-diffusion-2-1-unclip"
# model = "stabilityai/stable-diffusion-2"

pipeline = DiffusionPipeline.from_pretrained(model)
# pipeline.to("cuda")

image = pipeline("sexy girl in a space ship, high resolution").images[0]

image.save("images/sexygirl_spaceship5.png")
