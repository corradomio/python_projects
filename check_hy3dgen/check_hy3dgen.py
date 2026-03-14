import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

print("compose pipeline")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini',
    use_safetensors=True,
    device='cuda'
)

# image="D:\\Dropbox\\Pictures\\Sony Vaio\\sony_vaio_UX_Premium_1.jpg"
# image="D:\\Dropbox\\Pictures\\Bianca Beauchamp\\Bianca_Beauchamp_latex_manga-351206.jpg"
# image="D:\\Dropbox\\Pictures\\Bianca Beauchamp\\bianca_beauchamp___montreal_canadiens___playoffs_by_wolverine103197_dejsx22-fullview.jpg"
# it generates just a parallelepiped
# image = "D:\\Dropbox\\Panorama\\1000_F_66348236_upbTvfzNFjIAm4XJqNUo3Gz4g8igW7TH.jpg"

print("generating mesh")
mesh = pipeline(
    image=image,
    num_inference_steps=30,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]

print(mesh)
mesh.show()
print("mesh generated")

