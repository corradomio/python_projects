from diffusers import StableDiffusionPipeline
import torch

model_id = "Envvi/Inkpunk-Diffusion"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/envvi-1.jpg")

prompt = "beautiful young girl, high resolution, nvinkpunk "
image = pipeline(prompt).images[0]
image.save("./images/envvi-2.jpg")
