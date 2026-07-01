#
# https://github.com/damo-cv/TransReID
# https://arxiv.org/abs/2102.04378
#

import os.path
from pathlib import Path
from typing import cast

import gdown
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from .model import make_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSREID_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRANSREID_WEIGHTS_URLS = {
    # Pretrained
    "jx_vit_base_p16_224-80ecf9dd.pth": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
    "vit_small_p16_224-15ec54c9.pth": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    "deit_small_distilled_patch16_224-649709d9.pth": "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
    "deit_base_distilled_patch16_224-df68dfff.pth": "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",

    # MSMT17
    "MSMT17/vit_base": "https://drive.google.com/file/d/1iF5JNPw9xi-rLY3Ri9EY-PFAkK6Vg_Pf/view?usp=sharing",
    "MSMT17/vit_transreid": "https://drive.google.com/file/d/1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK/view?usp=sharing",
    "MSMT17/deit_transreid_stride": "https://drive.google.com/file/d/1WSUD0gKjGIG_gzTc2izH_y-EuDzweN95/view?usp=sharing",

    # Market
    "Market/vit_base": "https://drive.google.com/file/d/1crYsKRrW4eUq6abT4KK8_atMLFsbq56W/view?usp=sharing",
    "Market/vit_transreid": "https://drive.google.com/file/d/11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO/view?usp=sharing",
    "Market/deit_transreid_stride": "https://drive.google.com/file/d/1cbUK2KozdPSoewzvF0ucFQnZ0yfZiu_H/view?usp=sharing",

    # Duke
    "DukeMTMC/vit_base": "https://drive.google.com/file/d/17GQqFuTleAZWLD92AtEd1c_dnTyZHl4k/view?usp=sharing",
    "DukeMTMC/vit_transreid": "https://drive.google.com/file/d/1BipxoqyThefQviJzuJIKtFJvNblIlPGN/view?usp=sharing",
    "DukeMTMC/deit_transreid_stride": "https://drive.google.com/file/d/1ltaX9zGFO31Wwwu47K9c4WTTBZVLdzLw/view?usp=sharing",

    # OCC_Duke
    "OCC_Duke/vit_base": "https://drive.google.com/file/d/1uHX5j7yepalN1EINdF9lzrT3iDWj-pr9/view?usp=sharing",
    "OCC_Duke/vit_transreid": "https://drive.google.com/file/d/1VJg4rTA43TCHkR9hTIBu8S2Sy1KiTnSJ/view?usp=sharing",
    "OCC_Duke/deit_transreid_stride": "https://drive.google.com/file/d/1YJkBiMb5oVBnO6GXYW3Y_hFkR-Pl5ikC/view?usp=sharing",
}

TRANSREID_MODEL_WEIGHTS_NAMES = {
    "DukeMTMC/deit_transreid_stride": "deit_transreid_duke.pth",
    "DukeMTMC/vit_base": "vit_base_duke.pth",
    # "DukeMTMC/vit_jpm": "",
    # "DukeMTMC/vit_sie": "",
    # "DukeMTMC/vit_transreid": "dukemtmc/vit_transreid_duke.pth",
    # "DukeMTMC/vit_transreid_384": "",
    # "DukeMTMC/vit_transreid_stride": "",
    # "DukeMTMC/vit_transreid_stride_384": "",

    "Market/deit_transreid_stride": "deit_transreid_market.pth",
    "Market/vit_base": "vit_base_market.pth",
    # "Market/vit_jpm": "",
    # "Market/vit_sie": "",
    # "Market/vit_transreid": "market/vit_transreid_market.pth",
    # "Market/vit_transreid_384": "",
    # "Market/vit_transreid_stride": "",
    # "Market/vit_transreid_stride_384": "",

    # "MSMT17/deit_small": "",
    # "MSMT17/deit_transreid_stride": "msmt17/deit_transreid_msmt.pth",
    # "MSMT17/vit_base": "msmt17/vit_base_msmt.pth",
    # "MSMT17/vit_jpm": "",
    # "MSMT17/vit_sie": "",
    # "MSMT17/vit_small": "",
    # "MSMT17/vit_transreid": "msmt17/vit_transreid_msmt.pth",
    # "MSMT17/vit_transreid_384": "",
    # "MSMT17/vit_transreid_stride": "",
    # "MSMT17/vit_transreid_stride_384": "",

    "OCC_Duke/deit_transreid_stride": "deit_transreid_occ_duke.pth",
    # "OCC_Duke/vit_base": "occ_duke/vit_base_occ_duke.pth",
    # "OCC_Duke/vit_jpm": "",
    # "OCC_Duke/vit_sie": "",
    # "OCC_Duke/vit_transreid": "occ_duke/vit_transreid_occ_duke.pth",
    # "OCC_Duke/vit_transreid_stride",

    # "ReID780/vit_transreid",
}


TRANSREID_MODEL_NAMES = [
    name.replace("/", "__")
    for name in TRANSREID_MODEL_WEIGHTS_NAMES.keys()
]


TRANSREID_CLASSES_CAMERAS_VIEWS = {
    #         pids, cams, views
    "DukeMTMC": [702, 8, 1],
    "OCC_Duke": [702, 8, 1],
    "Market":   [751, 6, 1],
    # AI values
    "MSMT17": [1041, 12, 2],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

TRANSREID_WEIGHTS_ROOT = ".transreid_weights"

TRANSREID_MODELS = {}


def _model_name(cfg_name: str) -> str:
    p = cfg_name.find('/')
    return cfg_name[:p]


def _name_of(path: str) -> str:
    path = path.replace("\\","/")
    p = path.rfind('/')
    return path[p+1:]


def _get_cfg(cfg_name):
    from .config import cfg

    cfg_file = Path(__file__).parent / f"configs/{cfg_name}.yml"
    cfg_file = str(cfg_file)
    cfg.merge_from_file(cfg_file)
    # cfg.freeze()
    return cfg


def _download_weights(model_name, weights_path):
    assert model_name in TRANSREID_WEIGHTS_URLS
    url = TRANSREID_WEIGHTS_URLS[model_name]
    print(f"transreid: downloading {model_name} from {url} and saved in {weights_path}")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, weights_path, quiet=True)
    pass


def _download_weights_by_request(weights_name: str , weights_path: Path):
    assert weights_name in TRANSREID_WEIGHTS_URLS
    url = TRANSREID_WEIGHTS_URLS[weights_name]
    print(f"transreid: downloading {weights_name} from {url} and saved in {weights_path}")
    response = requests.get(url)
    response.raise_for_status()
    with open(weights_path, "wb") as file:
        file.write(response.content)
    pass


def _get_pretrained_weights_path(cfg_name, cfg) -> str:
    weights_name = _name_of(cfg.MODEL.PRETRAIN_PATH)
    weights_path = Path(TRANSREID_WEIGHTS_ROOT) / weights_name
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(weights_path):
        _download_weights_by_request(weights_name, weights_path)

    assert os.path.exists(weights_path)
    cfg.MODEL.PRETRAIN_PATH = weights_path
    return str(weights_path)


def _get_model_weights_path(cfg_name, cfg) -> str:
    weights_name = _name_of(TRANSREID_MODEL_WEIGHTS_NAMES[cfg_name])
    weights_path = Path(TRANSREID_WEIGHTS_ROOT) / weights_name

    if not weights_path.exists():
        _download_weights(cfg_name, weights_path)

    assert weights_path.exists()
    return str(weights_path)


def _get_model(cfg_name: str):
    global TRANSREID_MODELS
    if cfg_name in TRANSREID_MODELS:
        return TRANSREID_MODELS[cfg_name]

    cfg = _get_cfg(cfg_name)

    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_classes, camera_num, view_num = TRANSREID_CLASSES_CAMERAS_VIEWS[_model_name(cfg_name)]

    _get_pretrained_weights_path(cfg_name, cfg)
    model_weights_path = _get_model_weights_path(cfg_name, cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    model.eval().to(TRANSREID_DEVICE)

    TRANSREID_MODELS[cfg_name] = (cfg, transforms, model, camera_num, view_num)
    return cfg, transforms, model, camera_num, view_num


# ---------------------------------------------------------------------------
# TransReID
# ---------------------------------------------------------------------------

class TransReID:

    @staticmethod
    def represent(image: str|Path|np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = image
            # image = cv2.imread(filename)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(filename).convert("RGB")
        elif isinstance(image, np.ndarray):
            array = cast(np.ndarray, image)
            image = Image.fromarray(array, mode="RGB")

        cfg_name = model_name.replace("__", "/")
        (cfg, transforms, model, camera_num, view_num) = _get_model(cfg_name)

        timage: torch.Tensor = transforms(image).to(TRANSREID_DEVICE)[None]
        tdevice = timage.device
        camera_num = torch.tensor(0, device=tdevice).reshape((1))
        view_num = torch.tensor(1, device=tdevice).reshape((1))

        features = model(timage, cam_label=camera_num, view_label=view_num)
        # features = F.normalize(features)
        features = features.cpu().data.numpy()

        # features: np.ndarray[1, len] -> np.ndarray[len]
        return features.reshape(-1)
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str|Path|np.ndarray):
        return TransReID.represent(image, self._model_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        global TRANSREID_MODELS
        keys = list(TRANSREID_MODELS.keys())
        for k in keys:
            (cfg, transforms, model, camera_num, view_num) = TRANSREID_MODELS[k]
            model.to("cpu")
        TRANSREID_MODELS.clear()
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
