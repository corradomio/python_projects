from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)

model_id = "Yntec/Voxel"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Top left: VoxelArt. anime little girl with blonde messy hair, school uniform skirt, nose blush, beautiful eyes, sitting on her desk, front view, solo, full body
#
# Top right: kreatif, low poly, isometric art, 3d art, high detail, artstation, concept art, behance, ray tracing, smooth, sharp focus, ethereal lighting
#
# Bottom left: voxel venec fox pet closeup in convenience store. VoxelArt pixel chibi disney pixar, Voxel style, the perfect hero, Masterpiece photography, in hero pose, carrying, villains lair background, hyperrealistic, award winning photography, intricate textures, soft lighting,
#
# Bottom right: VoxelArt,1girl,long gray hair,girl sitting inside a long crystal bottle,bottle with stopper,water in bottle,grass, background,extremely detailed,natural lighting,film grain,

prompt = "kreatif, low poly, isometric art, 3d art, high detail, artstation, concept art, behance, ray tracing, smooth, sharp focus, ethereal lighting"
image = pipe(prompt).images[0]

image.save("./voxel-3.png")
