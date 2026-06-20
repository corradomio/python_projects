from PIL import Image
import requests

img_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")

import torch
from transformers import pipeline
from accelerate import Accelerator
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
device = Accelerator().device
pipe = pipeline(task="image-feature-extraction", model="google/vit-base-patch16-384", device=device, pool=True)

outputs = pipe([image_real, image_gen])

# get the length of a single output
print(len(outputs[0][0]))
# show outputs
print(outputs)

from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(torch.Tensor(outputs[0]),
                                     torch.Tensor(outputs[1]), dim=1)

print(similarity_score)
