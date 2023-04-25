# import torch
#
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
#
# model_id = "stabilityai/stable-diffusion-2-1"
#
# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
#
# image.save("astronaut_rides_horse.png")

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.enable_attention_slicing()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# image.save("images/astronaut_rides_horse.png")

# prompt = "portrait photo of a old warrior chief"
# image = pipe(prompt).images[0]
# image.save("images/portrait_old_warrior_chief.png")

# prompt = "girl astronaut in a spaceship"
# image = pipe(prompt).images[0]
# image.save("images/sexy_astronaut.png")

# prompt="Lamborghini Aventador LP 700-4 racing on a night road in Singapore, Asphalt 7: Heat, video game, london race, city light, cinematic lighting, night sky, sharp, digital painting, artstation, highly detailed, high resolution, uhd, 4 k, 8k wallpaper " \
#        "Negative prompt: ((dark)) duplicated blurry low-res haze (broken wheels)"
# image = pipe(prompt).images[0]
# image.save("images/lamborghini.png")

prompt="a portrait of a beautiful young female wearing an alexander mcqueen armor made of ice, photographed by andrew thomas huang, artistic, intricate drawing, light brazen, realistic fantasy, extremely detailed and beautiful aesthetic face, 8 k resolution, dramatic lighting, cinematic, dramatic lighting, masterpiece, trending on artstation, concept art, detailed face, high quality,"
image = pipe(prompt).images[0]
image.save("images/female_armor.png")

