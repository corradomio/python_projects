from diffusers import StableDiffusionPipeline
import torch

model_id = "nitrosocke/Future-Diffusion"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/nitrosocke-1.jpg")



model_id = "nitrosocke/nitrosocke/Arcane-Diffusion"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/nitrosocke-2.jpg")


