from diffusers import StableDiffusionPipeline
import torch

# -- not working
# model_id = "dreamlike-art/kandinsky-2.1"
# model_id = "ai-forever/Kandinsky_2.1"
# model_id = "ckpt/Kandinsky_2.1"
# -- end

# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
model_id = "dreamlike-art/dreamlike-anime-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting"
image = pipe(prompt).images[0]

image.save("images/church.jpg")
