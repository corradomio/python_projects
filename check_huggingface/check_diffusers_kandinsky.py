from diffusers import StableDiffusionPipeline
import torch

# -- not working
# model_id = "dreamlike-art/kandinsky-2.1"
# model_id = "ai-forever/Kandinsky_2.1"
# model_id = "ckpt/Kandinsky_2.1"
# model_id = "axolotron/ice-cream-animals"  # genera un'immagine NERA
# model_id = "ai-forever/Kandinsky_2.1"
# model_id = "skrashevich/kandinsky-2.0"
# -- end


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a beautiful perfect face girl in dgs illustration style, Anime fine details portrait of school girl in " \
         "front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high " \
         "quality anime"
image = pipe(prompt).images[0]
image.save("images/kandinsky_anime_girl_2.png")
