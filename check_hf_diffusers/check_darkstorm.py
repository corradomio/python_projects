from diffusers import StableDiffusionPipeline
import torch

model_id = "darkstorm2150/Protogen_x5.8_Official_Release"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "beautiful young girl, high resolution"
image = pipeline(prompt).images[0]
image.save("./images/darkstorm-1.jpg")
