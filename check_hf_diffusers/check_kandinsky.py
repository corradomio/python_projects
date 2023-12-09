from diffusers import AutoPipelineForText2Image
import torch

model = "kandinsky-community/kandinsky-3"
device = "cpu"

pipe = AutoPipelineForText2Image.from_pretrained(
    model, variant="fp16", torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
# pipe.enable_model_cuda_offload()


# prompt = ("A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is "
#           "reading a newspaper. The window shows the city in the background.")
prompt = ("Car, mustang, movie, person, poster, car cover, person, in the style of alessandro gottardo, gold and "
          "cyan, gerald harvey jones, reflections, highly detailed illustrations, industrial urban scenes")

generator = torch.Generator(device=device).manual_seed(0)
image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]


image.save("images/kandinsky-1.png")
