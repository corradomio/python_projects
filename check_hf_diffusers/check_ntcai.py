# https://huggingface.co/ntc-ai/SDXL-LoRA-slider.huge-anime-eyes
# https://huggingface.co/ntc-ai/SDXL-LoRA-slider.psychedelic-trip
# https://huggingface.co/ntc-ai/SDXL-LoRA-slider.looking-at-viewer

from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors")
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the LoRA
pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.looking-at-viewer', weight_name='looking at viewer.safetensors',
                       adapter_name="looking at viewer")

# Activate the LoRA
pipe.set_adapters(["looking at viewer"], adapter_weights=[2.0])

prompt = "medieval rich queenpin sitting in a tavern, looking at viewer"
negative_prompt = "nsfw"
width = 512
height = 512
num_inference_steps = 10
guidance_scale = 2
image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, guidance_scale=guidance_scale,
             num_inference_steps=num_inference_steps).images[0]
image.save('images/ntcai-looking-at-viewer.png')
