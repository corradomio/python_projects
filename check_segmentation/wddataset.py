from functools import lru_cache

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset

from stdlib.tprint import tprint

def to_classes(iclasses: np.ndarray, n_classes, classes_last=True) -> np.ndarray:
    pclasses = np.zeros(iclasses.shape + (n_classes,), dtype=np.float32)
    for c in range(n_classes):
        pclasses[iclasses == c, c] = 1.
    if not classes_last:
        pclasses = np.moveaxis(pclasses, 0, 2)
    return pclasses


class WaterDropDataset(Dataset):

    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDEV = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, homedir: str, start: int=0, count: int=1000,
                 channels_last=True,
                 classes_last=True):
        self.homedir = homedir
        self.start = start
        self.count = count
        self.channels_last=channels_last
        self.classes_last=classes_last

    def __len__(self):
        return self.count

    @lru_cache(1000)
    def __getitem__(self, item):
        iimage = self.start + item
        idir = iimage//1000

        tprint(f"... loading item {iimage:04}")

        scene_file = f"{self.homedir}/{idir:04}/drop_scene-{iimage:04}.png"
        mask_file  = f"{self.homedir}/{idir:04}/drop_mask-{iimage:04}.png"

        # drop_scene (96, 96, 3) uint8
        drop_scene: np.ndarray = iio.imread(scene_file)
        drop_scene = (drop_scene / 255.).astype(np.float32)
        drop_scene = (drop_scene - self.MEANS)/self.STDEV
        if not self.channels_last:
            drop_scene = np.moveaxis(drop_scene, -1, 0)

        # drop_mask (96, 96)
        # classes: [1,4] -> [0,3]
        drop_mask: np.ndarray = iio.imread(mask_file)-1
        drop_mask = drop_mask.astype(np.int64)
        drop_mask = to_classes(drop_mask, 3, classes_last=self.classes_last)

        drop_scene_t = torch.from_numpy(drop_scene).contiguous()
        drop_mask_t  = torch.from_numpy(drop_mask).contiguous()

        return drop_scene_t, drop_mask_t
    # end

    def get_image(self, item):
        iimage = self.start + item
        idir = iimage // 1000

        tprint(f"... loading item {iimage:04}")

        scene_file = f"{self.homedir}/{idir:04}/drop_scene-{iimage:04}.png"
        mask_file  = f"{self.homedir}/{idir:04}/drop_mask-{iimage:04}.png"

        # drop_scene
        drop_scene: np.ndarray = iio.imread(scene_file)
        drop_scene = (drop_scene / 255.).astype(np.float32)
        drop_scene = (drop_scene - self.MEANS) / self.STDEV
        # [W,H,C] -> [C,W,H]
        # drop_scene = np.moveaxis(drop_scene, -1, 0)
        drop_scene = drop_scene.reshape((1,) + drop_scene.shape)

        # drop_mask (96, 96)
        # classes: [1,4] -> [0,3]
        drop_mask: np.ndarray = iio.imread(mask_file)-1
        drop_mask = drop_mask.reshape((1,) + drop_mask.shape)
        # drop_mask = drop_mask/255.

        drop_scene_t = torch.from_numpy(drop_scene).contiguous()
        # drop_mask_t  = torch.from_numpy(drop_mask).contiguous()

        return drop_scene_t, drop_mask
    # end
# end