from diffusers import StableDiffusionPipeline
import torch

model_id = "plasmo/vox2"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-1.jpg")

prompt = "beautiful young girl, high resolution, voxel-ish"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-2.jpg")
