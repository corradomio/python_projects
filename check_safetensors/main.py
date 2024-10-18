from safetensors import *
from stdlib.dict import dict

FILE = "E:/Huggingface/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/unet/diffusion_pytorch_model.safetensors"

def main():
    tensors = dict()

    with safe_open(FILE, framework="pt", device=0) as f:
        print(type(f))
        for k in f.keys():
            # tensors.set(k, f.get_tensor(k))
            # print(k, tensors.get(k).shape)
            tensors[k] = f.get_tensor(k)
            print(k, tensors[k].shape)
    pass


if __name__ == '__main__':
    main()
