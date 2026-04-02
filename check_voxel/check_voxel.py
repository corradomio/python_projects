import matplotlib.pyplot as plt
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('Hardel-DW/voxelia', weight_name='lora.safetensors')
image = pipeline('medioeval castle near a lake').images[0]


plt.imsave("castle.png")
