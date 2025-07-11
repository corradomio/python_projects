import imageio.v3 as iio
import numpy as np
import torch
import matplotlib.pyplot as plt
from path import Path as path

from mask_classes import mask_classes
from torchx.nn import SegNet

IMAGES_DIR = r"E:\Datasets\WaterDrop\orig"
# IMAGES_DIR = r"images"


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


segnet = SegNet(in_channels=1, num_classes=4)
# summary(unet, input_size=(1,640,480, 1))


def main():
    images_dir = path(IMAGES_DIR)

    scene_file = images_dir / "0000/drop_scene-0000.png"
    mask_file  = images_dir / "0000/drop_mask-0000.png"

    scene: np.ndarray = iio.imread(scene_file)
    plt.imshow(scene)
    plt.show()
    # (480, 640, 3)
    scene = scene[:,:,:1]
    # (480, 640, 1)
    scene = scene.astype(np.float32)/255.
    # (480, 640)
    scene_mask: np.ndarray = iio.imread(mask_file)
    scene_classes = mask_classes(scene_mask).astype(np.float32)

    X = scene.reshape((1,) + scene.shape)
    X = torch.from_numpy(X)
    y = scene_classes.reshape((1,) + scene_classes.shape)
    y = torch.from_numpy(y)

    # (B, C, H, W)
    ret = segnet.forward(X)
    pred = ret[0].detach().numpy().transpose(1, 2, 0)
    plt.imshow(pred)
    plt.show()
    pass


if __name__ == "__main__":
    main()
