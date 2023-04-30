# https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion
import torch
from diffusers import StableDiffusionPipeline


model_id = "DGSpitzer/Cyberpunk-Anime-Diffusion"
# model_id = "xyn-ai/Cyberpunk-Anime-Diffusion"
# model_id = "MirageML/lowpoly-cyberpunk"
# model_id = "AdamOswald1/Cyberpunk-Anime-Diffusion_with_support_for_Gen-Imp_characters"
# model_id = "AdamOswald1/Cyberpunk-Anime-Diffusion"
# model_id = "flax/Cyberpunk-Anime-Diffusion"
# model_id = ""

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a beautiful perfect face girl in dgs illustration style, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high quality anime"
image = pipe(prompt).images[0]

image.save("./images/cyberpunk_girl_2.png")
