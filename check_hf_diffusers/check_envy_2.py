from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('e-n-v-y/envy-reclaimed-brutalism-xl-01', weight_name='EnvyReclaimedBrutalismXL01.safetensors')


prompts = [
    'reclaimed brutalism, colorful, open markets,colorful banners, noon, blue sky, clouds, scenery, "at the Incandescent Crystal"'
    'colorful, fantasy, cheerful,great fantasy megacity at the end of time',
    'colorful, ivy,colorful banners, anime style, 1boy, man, ruggedly handsome, whimsical,Dystopian fantasy megastructure beyond the beginning of time',
    'reclaimed brutalism, colorful, open markets,colorful banners, noon, blue sky, clouds, scenery, "at the Incandescent Crystal"',
    'colorful, sakura trees, a demonic scifi city edge of the multiverse, masterpiece, by Vitaly Bulgarov',
    'reclaimed brutalism, colorful, happy crowds,open markets, anime style, 1girl, woman, beautiful, great smeltery in a far future fantasy topia outside of the universe, masterpiece, by Dmitry Vishnevsky',
    'reclaimed brutalism, colorful, people, warmly lit interior, in a adirondack Renaissance fair marketplace',
    'reclaimed brutalism, colorful, lake, anime style, 1boy, man, ruggedly handsome, golden hour, blue sky, clouds, scenery, in a Mudflat',
    'reclaimed brutalism, colorful, well-lit interior, in a Homely Golden wheat field at dusk'
]

id = 0
for p in prompts:
    id += 1
    image = pipeline(
        p
    ).images[0]
    image.save(f"images/envy-2-{id}.png")


