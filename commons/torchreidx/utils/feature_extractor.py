import os
import numpy as np
import torch
import torchreid.utils.feature_extractor as tidufe
from path import Path


INPUT_TYPE = str | list[str] | np.ndarray | torch.Tensor

def filter_ext(files: list[str], ext: str|list[str]|tuple[str]) -> list[str]:
    if isinstance(ext, str):
        ext = [ext]
    if isinstance(ext, list):
        ext = tuple(ext)
    filtered = []
    for f in files:
        if f.endswith(ext):
            filtered.append(f)
    return filtered


class FeatureExtractor(tidufe.FeatureExtractor):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        super(FeatureExtractor, self).__init__(
            model_name=model_name,
            model_path=model_path,
            image_size=image_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            device=device,
            verbose=verbose
        )

    def __call__(self, input: INPUT_TYPE, ext=('.jpg', '.png', '.jpeg')):
        # list | str | np.ndarray | tprch.Tensor
        if (isinstance(input, str)) and os.path.isdir(input):
            dir = input
            input = os.listdir(dir)
            input = [(f"{dir}/{f}") for f in input if f.endswith(ext)]
        return super().__call__(input)

    def extract(self, input: INPUT_TYPE, ext=('.jpg', '.png', '.jpeg')):
        return self.__call__(input, ext)
