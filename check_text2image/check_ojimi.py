from diffusers import DiffusionPipeline
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)

pipe = DiffusionPipeline.from_pretrained("Ojimi/anime-kawai-diffusion")
pipe = pipe.to("cuda")

prompt = "1girl, animal ears, long hair, solo, cat ears, choker, bare shoulders, red eyes, fang, looking at viewer, animal ear fluff, upper body, black hair, blush, closed mouth, off shoulder, bangs, bow, collarbone"
image = pipe(prompt, negative_prompt="lowres, bad anatomy").images[0]


image.save("./ojimi.png")
