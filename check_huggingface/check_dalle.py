#
# https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb
# https://colab.research.google.com/drive/1Blh-hTfhyry-YvitH8A95Duzwtm17Xz-?usp=sharing
#
from diffusers import StableDiffusionPipeline
import torch

# -- not working
# model_id = "dreamlike-art/kandinsky-2.1"
# model_id = "ai-forever/Kandinsky_2.1"
# model_id = "ckpt/Kandinsky_2.1"
# -- end

# model_id = "dalle-mini/dalle-mega"
# model_id = "dalle-mini/dalle-mini"
# model_id = "nev/dalle-mini-pytorch"
# model_id =
model_id = "dalle2/dreamweddingbooth"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting"
image = pipe(prompt).images[0]

image.save("images/church.jpg")

