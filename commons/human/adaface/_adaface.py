from pathlib import Path

import cv2
import gdown
import numpy as np
import torch

from . import net

#
# R18
#     CASIA-WebFace       https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing
#     VGGFace2            https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing
#     WebFace4M           https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing
#
# R50
#     CASIA-WebFace       https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing
#     WebFace4M           https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing
#     MS1MV2              https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing
#
# R100
#     MS1MV2              https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing
#     MS1MV3              https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing
#     WebFace4M           https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing
#     WebFace12M          https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAFACE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADAFACE_WEIGHTS_URLS = {
    "R18-CASIA-WebFace" : "https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing",
    "R18-VGGFace2"      : "https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing",
    "R18-WebFace4M"     : "https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing",

    "R50-CASIA-WebFace" : "https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing",
    "R50-WebFace4M"     : "https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing",
    "R50-MS1MV2"        : "https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing",

    "R100-MS1MV2"       : "https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing",
    "R100-MS1MV3"       : "https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing",
    "R100-WebFace4M"    : "https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing",
    "R100-WebFace12M"   : "https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing",
}

ADAFACE_MODEL_WEIGHTS_NAMES = {
    "R18-CASIA-WebFace": "adaface_ir18_casia.ckpt",
    "R18-VGGFace2": "adaface_ir18_vgg2.ckpt ",
    "R18-WebFace4M": "adaface_ir18_webface4m.ckpt",

    "R50-CASIA-WebFace": "adaface_ir50_casia.ckpt",
    "R50-WebFace4M": "adaface_ir50_webface4m.ckpt",
    "R50-MS1MV2": "adaface_ir50_ms1mv2.ckpt",

    "R100-MS1MV2": "adaface_ir101_ms1mv2.ckpt",
    "R100-MS1MV3": "adaface_ir101_ms1mv3.ckpt",
    "R100-WebFace4M": "adaface_ir101_webface4m.ckpt",
    "R100-WebFace12M": "adaface_ir101_webface12m.ckpt",
}


ADAFACE_MODEL_NAMES = [
    name.replace("/", "__")
    for name in ADAFACE_MODEL_WEIGHTS_NAMES.keys()
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ADAFACE_WEIGHTS_ROOT = ".adaface_weights"

ADAFACE_MODELS = {}

def _arch_of(model_name: str) -> str:
    if model_name.startswith("R18-"):
        return "ir_18"
    if model_name.startswith("R50-"):
        return "ir_50"
    if model_name.startswith("R100-"):
        return "ir_101"
    if model_name.startswith("R101-"):
        return "ir_101"
    raise ValueError(f"Unknown architecture: {model_name}")


def _download_weights(model_name, weights_path):
    assert model_name in ADAFACE_WEIGHTS_URLS
    url = ADAFACE_WEIGHTS_URLS[model_name]
    print(f"adaface: downloading {model_name} from {url} and saved in {weights_path}")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(weights_path), quiet=True)
    pass


def _get_model_weights_path(model_name: str) -> str:
    weights_name = ADAFACE_MODEL_WEIGHTS_NAMES[model_name]
    weights_path = Path(ADAFACE_WEIGHTS_ROOT) / weights_name

    if not weights_path.exists():
        _download_weights(model_name, weights_path)

    assert weights_path.exists()
    return str(weights_path)


def _get_model(model_name):
    if model_name in ADAFACE_MODELS:
        return ADAFACE_MODELS[model_name]

    assert model_name in ADAFACE_MODEL_NAMES

    weights_path = _get_model_weights_path(model_name)

    statedict = torch.load(weights_path, weights_only=False)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}

    architecture = _arch_of(model_name)
    model = net.build_model(architecture)

    model.load_state_dict(model_statedict)
    model.eval().to(ADAFACE_DEVICE)

    ADAFACE_MODELS[model_name] = model
    return model
# end


def to_input(np_img: np.ndarray) -> torch.Tensor:
    assert isinstance(np_img, np.ndarray)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(brg_img.transpose(2,0,1).reshape((1, -1))).float().to(ADAFACE_DEVICE)
    return tensor


# ---------------------------------------------------------------------------
# InsightFaceReID
# ---------------------------------------------------------------------------
# WARNING:
# Note that our pretrained model takes the input in BGR color channel.
# This is different from the InsightFace released model which uses RGB color channel.

class AdaFace:

    @staticmethod
    def represent(image: str | Path | np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = str(image)
            image = cv2.imread(filename)

            # NOT TO CONVERT IN RGB!!!
            # To see: https://github.com/mk-minchul/adaface
            # in section 'Pretrained Models'
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.open(filename).convert("RGB")
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        model = _get_model(model_name)

        resized_image = cv2.resize(image, (112, 112))
        bgr_input = to_input(resized_image)

        with torch.no_grad():
            feature, _ = model(bgr_input)
            feature = feature.detach().cpu().numpy()

        return feature
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return AdaFace.represent(image, self._model_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        keys = list(ADAFACE_MODELS.keys())
        for k in keys:
            ADAFACE_MODELS[k].to("cpu")
        ADAFACE_MODELS.clear()
        pass
    # end
# end
