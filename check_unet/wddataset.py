from functools import lru_cache

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset

from stdlib.tprint import tprint


class WaterDropDataset(Dataset):

    def __init__(self, homedir: str, start: int=0, count: int=8000):
        self.homedir = homedir
        self.start = start
        self.count = count

    def __len__(self):
        return self.count

    @lru_cache(2500)
    def __getitem__(self, item):
        iimage = self.start + item
        idir = iimage//1000

        tprint(f"... loading item {iimage:04}")

        scene_file = f"{self.homedir}/{idir:04}/drop_scene-{iimage:04}.png"
        mask_file  = f"{self.homedir}/{idir:04}/drop_mask-{iimage:04}.png"

        # drop_scene
        drop_scene: np.ndarray = iio.imread(scene_file).astype(np.float32)[:, :, :1]
        h, w, c = drop_scene.shape
        # drop_scene = resize(drop_scene, (h//2, w//2))
        drop_scene = drop_scene / 255.

        # drop_mask
        # classes: 1-4 -> [0,3]
        drop_mask: np.ndarray = iio.imread(mask_file).astype(np.int64)
        # drop_mask = resize(drop_mask, (h // 2, w // 2))
        drop_mask -= 1

        return torch.from_numpy(drop_scene), torch.from_numpy(drop_mask)
    # end

    def get_image(self, item):
        iimage = self.start + item
        idir = iimage // 1000

        tprint(f"... loading item {iimage:04}")


        scene_file = f"{self.homedir}/{idir:04}/drop_scene-{iimage:04}.png"
        mask_file  = f"{self.homedir}/{idir:04}/drop_mask-{iimage:04}.png"

        # drop_scene
        drop_scene: np.ndarray = iio.imread(scene_file).astype(np.float32)[:, :, :1]
        drop_scene = drop_scene / 255.
        drop_scene = drop_scene.reshape((1,) + drop_scene.shape)

        # drop_mask
        # classes: 1-4 -> [0,3]
        drop_mask: np.ndarray = iio.imread(mask_file).astype(np.int64)
        drop_mask -= 1
        drop_mask = drop_mask.reshape((1,) + drop_mask.shape)

        return torch.from_numpy(drop_scene), torch.from_numpy(drop_mask)
    # end
# end