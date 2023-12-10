from diffusers import StableDiffusionPipeline
import torch

model = "plasmo/vox2"
# model = "plasmo/voxel-ish"

pipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "two lions sit down in the savanna, high resolution, voxel-ish"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-2.1.jpg")
