from diffusers import StableDiffusionPipeline
import torch

# -- not working
# model_id = "dreamlike-art/kandinsky-2.1"
# model_id = "ai-forever/Kandinsky_2.1"
# model_id = "ckpt/Kandinsky_2.1"
# -- end

# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# model_id = "dreamlike-art/dreamlike-anime-1.0"
# model_id = "Duskfallcrew/duskfall-s-manga-aesthetic-model"
model_id = "xyn-ai/Cyberpunk-Anime-Diffusion"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting"
# image = pipe(prompt).images[0]
# image.save("images/church.jpg")

prompt = "a beautiful perfect face girl in dgs illustration style, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high quality anime"
image = pipe(prompt).images[0]
image.save("images/cyberpunk_girl.png")
