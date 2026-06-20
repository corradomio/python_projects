from typing import cast

import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from .models import build_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TORCHREID_MODEL_NAMES = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512',
    'se_resnet50', 'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
    'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512',
    'inceptionresnetv2', 'inceptionv4', 'xception',
    'resnet50_ibn_a', 'resnet50_ibn_b', 'nasnsetmobile',
    'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet',
    'squeezenet1_0', 'squeezenet1_0_fc512', 'squeezenet1_1',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'mudeep', 'resnet50mid',
    # 'hacnn', # (h:160, w:64)
    'pcb_p6', 'pcb_p4', 'mlfn',
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
    'osnet_ibn_x1_0',
    'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25'
]


TORCHREID_MODELS: dict[str, nn.Module] = {}

TORCHREID_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_model(model_name):
    if model_name in TORCHREID_MODELS:
        return TORCHREID_MODELS[model_name]

    model = build_model(
        name=model_name,
        num_classes=1000,
        pretrained=True
    )

    model.eval().to(TORCHREID_DEVICE)
    TORCHREID_MODELS[model_name] = model
    return model


FEATURE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# TorchReID
# ---------------------------------------------------------------------------

class TorchReID:

    @staticmethod
    def represent(image: str|Path|np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = image
            image: np.ndarray = cast(np.ndarray, cv2.imread(str(filename)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = FEATURE_TRANSFORMS(image).unsqueeze(0).to(TORCHREID_DEVICE)

        model = _get_model(model_name)
        with torch.no_grad():
            features = model(image)

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()[0]
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return TorchReID.represent(image, self._model_name)
# end



# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
