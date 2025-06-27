import imageio.v3 as iio
import numpy as np
import torch
from path import Path as path

from mask_classes import mask_classes
from torchx.nn import UNet

IMAGES_DIR = path(r"E:\Datasets\WaterDrop\cropped")

unet = UNet(in_channels=3, num_classes=4, channels_last=False)


def main():
    scene_file = IMAGES_DIR / "0000/drop_scene-0000.png"
    mask_file  = IMAGES_DIR / "0000/drop_mask-0000.png"

    scene: np.ndarray = iio.imread(scene_file)[:,:,:1]
    scene = scene.astype(np.float32)/255.
    # (480, 640, 3)
    # (480, 640, 1)
    # (480, 640)
    scene_mask: np.ndarray = iio.imread(mask_file)
    scene_classes = mask_classes(scene_mask).astype(np.float32)

    X = scene.reshape((1,) + scene.shape)
    X = torch.from_numpy(X)
    y = scene_classes.reshape((1,) + scene_classes.shape)
    y = torch.from_numpy(y)

    print(X.shape)

    Y = unet.forward(X)

    print(Y.shape)

    # for dir in images_dir.dirs():
    #     tprint(dir, force=True)
    #     for scene_file in dir.files("drop_scene-*.png"):
    #         idir: path = scene_file.parent
    #
    #         scene_id = idof(scene_file.stem)
    #         mask_file = idir / f"drop_mask-{scene_id}.png"
    #
    #         scene: np.ndarray = iio.imread(scene_file)
    #         # (480, 640, 3)
    #         # (480, 640, 1)
    #         # (480, 640)
    #         scene_mask: np.ndarray = iio.imread(mask_file)
    #         scene_classes = mask_classes(scene_mask)
    #
    #
    #         X = scene.reshape((1,) + scene.shape)
    #         X = torch.from_numpy(X)
    #         y = scene_classes.reshape((1,) + scene_classes.shape)
    #         y = torch.from_numpy(y)
    #
    #         ret = unet.forward(X)
    #
    #         # net.fit(X, y)





if __name__ == "__main__":
    main()
