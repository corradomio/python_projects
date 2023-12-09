from diffusers import AutoPipelineForText2Image
import torch

model = "stabilityai/sd-turbo"
# model = "stabilityai/sdxl-turbo"


pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
prompt = "beautiful young girl, high resolution"

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

image.save("images/stabilityai-2.1.png")
