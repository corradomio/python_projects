import os
import torch
import transformers
from diffusers import StableDiffusionPipeline

os.environ["PYTORCH_CUDA_ALLOC_CON"] = "max_split_size_mb:14336"

model = "prompthero/openjourney"

pipeline = StableDiffusionPipeline.from_pretrained(model)
pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("images/prompthero-1.png")
