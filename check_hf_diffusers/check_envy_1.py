from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('e-n-v-y/envy-arid-modernism-xl-01', weight_name='EnvyAridModernismXL01.safetensors')

prompts = [
    'anime style, 1girl, woman, beautiful, cheerful,indescribable scifi city beyond the end of the multiverse',
    "i pond in a city park in a great fantasy cloud metropolis beyond the end of reality, masterpiece",
    'anime style, 1girl, woman, beautiful, golden hour, blue sky, clouds, scenery, "at the indescribable Symposium"',
    'anime style, 1boy, man, ruggedly handsome, quantum singularity reactor in a far future scifi vertical topia beyond the end of the multiverse, masterpiece',
    'a empty fantasy city outside of the universe, masterpiece',
    'scifi, Alpine Tundra',
    'anime style, 1boy, man, ruggedly handsome, noon, blue sky, clouds, scenery, in a Lonely Buried Treasure Islands',
    'fantasy, a Radiant,amazing fantasy cloud megacity outside of reality, masterpiece',
    'anime style, 1boy, man, ruggedly handsome, golden hour, scenery, in a Enigmatic Cave'
]

id = 0
for p in prompts:
    id += 1
    image = pipeline(
        p
    ).images[0]
    image.save(f"images/envy-1-{id}.png")

