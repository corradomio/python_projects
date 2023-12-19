from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('e-n-v-y/envy-technobrutalist-xl-01', weight_name='EnvyTechnobrutalistXL01.safetensors')


prompts = [
    'technobrutalist, 1girl, woman, rough digital painting, (full body:1.2), 1boy, man, masculine, solo, [:outlandish costume design,:0.2] priest, crimson (ornate brass pauldrons,shield:1.4), caucasian, neon slate blue hair, (average:1) build, old, simple background, moba character concept art, bombshell hair, glowing yellow hair with aqua highlights, side braid, toned body, athletic build, narrow waist, small breasts, caucasian',
    'technobrutalist, 1girl, woman, rough digital painting, (full body:1.2), 1boy, man, masculine, solo, [:outlandish costume design,:0.2] priest, crimson (ornate brass pauldrons,shield:1.4), caucasian, neon slate blue hair, (average:1) build, old, simple background, moba character concept art, bombshell hair, glowing yellow hair with aqua highlights, side braid, toned body, athletic build, narrow waist, small breasts, caucasian'
]


id = 0
for p in prompts:
    id += 1
    image = pipeline(
        p
    ).images[0]
    image.save(f"images/envy-3-{id}.png")
