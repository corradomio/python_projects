from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)

model_id = "Fictiverse/Stable_Diffusion_PaperCut_Model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "PaperCut R2-D2"
image = pipe(prompt).images[0]

image.save("./R2-D2.png")
