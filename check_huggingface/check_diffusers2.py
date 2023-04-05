# import torch
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
#
# model_id = "stabilityai/stable-diffusion-2-1"
#
# # Use the Euler scheduler here instead
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "An image of a young sexy naked woman, colored, high resolution"
# images = pipe(prompt).images
# for i in range(len(images)):
#     images[i].save(f"images/naked-woman.png")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
print(torch.__version__)


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "An image of a young sexy naked woman, colored, medium resolution"
images = pipe(prompt).images
for i in range(len(images)):
    images[i].save(f"images/naked-woman-{i}.png")
