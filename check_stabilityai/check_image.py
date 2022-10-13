# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
from torch import autocast
from diffusers import StableDiffusionPipeline
from huggingface_hub import notebook_login

# access_token="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"
access_token="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"

pipe = StableDiffusionPipeline.from_pretrained(
    # "CompVis/stable-diffusion",
    "CompVis/stable-diffusion-v1-4"
    , se_auth_token=access_token
    # , use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
