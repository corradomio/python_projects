from diffusers import StableDiffusionPipeline
import torch

model = "plasmo/vox2"
# model = "plasmo/voxel-ish"

pipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution, intricate detail, voxel-ish"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-1.1.jpg")

prompt = "wizard, high resolution, intricate detail, voxel-ish"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-wizard-1.2.jpg")

prompt = "wonder woman, high resolution, intricate detail, voxel-ish"
image = pipeline(prompt).images[0]
image.save("./images/plasmo-ww-1.3.jpg")
