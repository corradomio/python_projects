from diffusers import DiffusionPipeline
import torch

# model = "playgroundai/playground-v1"
model = "playgroundai/playground-v2-1024px-aesthetic"
# model = "playgroundai/playground-v2-1024px-base"
# model = "playgroundai/playground-v2-512px-base"
# model = "playgroundai/playground-v2-256px-base"


pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16",
)
pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt, width=512, height=512).images[0]


image.save("images/playgroundai-1.png")
