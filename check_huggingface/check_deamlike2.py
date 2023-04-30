from diffusers import StableDiffusionPipeline
import torch

model_id = "dreamlike-art/dreamlike-anime-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, bangs, blue " \
         "eyes, skirt, huge breasts, iridescent, gradient, colorful, besides a cottage, in the country"
negative_prompt = 'simple background, duplicate, retro style, low quality, lowest quality, ' \
                  '1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, ' \
                  'bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, ' \
                  'watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry'
image = pipe(prompt, negative_prompt=negative_prompt).images[0]

image.save("./images/anime_girl_4.jpg")
