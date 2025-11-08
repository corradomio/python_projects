import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from huggingface_hub import login
from stdlib.jsonx import load

token = load(r"D:\Projects.github\python_projects\api_key_tokens.json")["hf"]
login(token=token)

pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a firefighter"
image = pipeline(
    caption,
    height=1024,
    width=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=4,
).images[0]


image.save("wuerstchen.png")
