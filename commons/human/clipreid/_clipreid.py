# https://github.com/Syliz517/CLIP-ReID
#

import os
from typing import cast

import torch
import numpy as np
import gdown
import torchvision.transforms as T
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIPREID_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLIPREID_WEIGHTS_URLS = {
    "Market/cnn_base": "https://drive.google.com/file/d/15E4K9eGXMlqOGE1RAgXQjF4MzrFobGim/view?usp=share_link",
    "Market/cnn_clipreid": "https://drive.google.com/file/d/1sBqCr5LxKcO9J2V0IvLQPb0wzwVzIZUp/view?usp=share_link",

    "MSMT17/cnn_base": "https://drive.google.com/file/d/1s-nZMp-LHG0h4dFwvyP_YNBLTijLcrb0/view?usp=share_link",
    "MSMT17/cnn_clipreid": "https://drive.google.com/file/d/1VdlC1ld3NrQC5Jcx0hntXRb-UaR3tMtr/view?usp=share_link",

    "DukeMTMC/cnn_base": "https://drive.google.com/file/d/1f9ZgJZSph7kV7xjhfBVIjFG0hwgeSsSy/view?usp=share_link",
    "DukeMTMC/cnn_clipreid": "https://drive.google.com/file/d/1XXycuux__uDd9WKwaTAQ4W1RjLqnUphq/view?usp=share_link",

    "OCC_Duke/cnn_base": "https://drive.google.com/file/d/1gdokL9QoldUOiaRUGJ1fS0BXEnHGM8MX/view?usp=share_link",
    "OCC_Duke/cnn_clipreid": "https://drive.google.com/file/d/1naz7QjzYlC2qe4SHxjxss4tP81KRCrMj/view?usp=share_link",
}


CLIPREID_MODEL_WEIGHTS_NAMES = {
    "Market/cnn_base": "Market1501_baseline_RN50_120.pth",
    "Market/cnn_clipreid": "Market1501_clipreid_RN50_120.pth",

    "MSMT17/cnn_base": "MSMT17_baseline_RN50_120.pth",
    "MSMT17/cnn_clipreid": "MSMT17_clipreid_RN50_120.pth",

    "DukeMTMC/cnn_base": "Duke_baseline_RN50_120.pth",
    "DukeMTMC/cnn_clipreid": "Duke_clipreid_RN50_120.pth",

    "OCC_Duke/cnn_base": "Occ_Duke_baseline_RN50_120.pth",
    "OCC_Duke/cnn_clipreid": "Occ_Duke_clipreid_RN50_120.pth",
}


CLIPREID_MODEL_NAMES = [
    name.replace("/", "__")
    for name in CLIPREID_MODEL_WEIGHTS_NAMES.keys()
]


CLIPREID_CLASSES_CAMERAS_VIEWS = {
    #         pids, cams, views
    "DukeMTMC": [702, 8, 1],
    "OCC_Duke": [702, 8, 1],
    "Market":   [751, 6, 1],
    # ???
    "MSMT17": [1041, 12, 2],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

CLIPREID_WEIGHTS_ROOT = ".clipreid_weights"

CLIPREID_MODELS = {}


def _model_name(cfg_name: str) -> str:
    p = cfg_name.find('/')
    return cfg_name[:p]


def _name_of(path: str) -> str:
    path = path.replace("\\","/")
    p = path.rfind('/')
    return path[p+1:]


def _get_cfg(cfg_name):
    if "cnn_base" in cfg_name:
        from .config import cfg_base as cfg

        cfg_file = Path(__file__).parent / f"configs/{cfg_name}.yml"
        cfg_file = str(cfg_file)
        cfg.merge_from_file(cfg_file)
    elif "cnn_clipreid" in cfg_name:
        from .config import cfg
        cfg_file = Path(__file__).parent / f"configs/{cfg_name}.yml"
        cfg_file = str(cfg_file)
        cfg.merge_from_file(cfg_file)
    else:
        raise ValueError(f"Unsupported {cfg_name}")

    cfg.freeze()
    return cfg


def _download_weights(model_name: str, weights_path: Path):
    assert model_name in CLIPREID_WEIGHTS_URLS
    url = CLIPREID_WEIGHTS_URLS[model_name]

    print(f"clipreid: downloading {model_name} from {url} and saved in {weights_path}")

    gdown.download(url, str(weights_path), quiet=True)
    pass


def _get_model_weights_path(cfg_name, cfg) -> str:
    weights_name = _name_of(CLIPREID_MODEL_WEIGHTS_NAMES[cfg_name])
    weights_path = Path(CLIPREID_WEIGHTS_ROOT) / weights_name
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(weights_path):
        _download_weights(cfg_name, weights_path)

    assert os.path.exists(weights_path)
    return str(weights_path)


def _get_model(cfg_name: str):
    global CLIPREID_MODELS
    if cfg_name in CLIPREID_MODELS:
        return CLIPREID_MODELS[cfg_name]

    cfg = _get_cfg(cfg_name)

    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_classes, camera_num, view_num = CLIPREID_CLASSES_CAMERAS_VIEWS[_model_name(cfg_name)]

    model_weights_path = _get_model_weights_path(cfg_name, cfg)

    if "cnn_base" in cfg_name:
        from .model.make_model import make_model
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    else:
        from .model.make_model_clipreid import make_model
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    model.load_param(model_weights_path)
    model.eval().to(CLIPREID_DEVICE)

    CLIPREID_MODELS[cfg_name] = (cfg, transforms, model, camera_num, view_num)
    return cfg, transforms, model, camera_num, view_num


# ---------------------------------------------------------------------------
# ClipReID
# ---------------------------------------------------------------------------


class ClipReID:

    @staticmethod
    def represent(image: str | Path | np.ndarray, model_name: str) -> np.ndarray:
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

        timage: torch.Tensor = transforms(image).to(CLIPREID_DEVICE)[None]
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

    def embedding(self, image: str | Path | np.ndarray):
        return ClipReID.represent(image, self._model_name)
# end
