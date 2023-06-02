from diffusers import StableDiffusionPipeline
import torch

model_id = "stablediffusionapi/rev-anim"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/stablediffusionapi-1.jpg")


model_id = "stablediffusionapi/edge-of-realism"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/stablediffusionapi-2.jpg")


model_id = "stablediffusionapi/cosmic-babes"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/stablediffusionapi-3.jpg")
