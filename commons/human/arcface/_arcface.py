#
# https://github.com/yakhyo/face-reidentification
#
import os
from pathlib import Path
from typing import cast

import requests
import torch
import numpy as np
import cv2
from .arcface import YakhyoArcFace
from .scrfd import YakhyoSCRFD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCFACE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCFACE_WEIGHTS_URLS = {
    # detector
    "det_500m.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx",
    "det_2.5g.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
    "det_10g.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx",

    # recognizer
    "w600k_mbf.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx",
    "w600k_r50.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx",
}


ARCFACE_MODEL_WEIGHTS_NAMES = {
    # detector, recognizer
    "det_500m-w600k_mbf": ["det_500m.onnx", "w600k_mbf.onnx"],
    "det_500m-w600k_r50": ["det_500m.onnx", "w600k_r50.onnx"],
    
    "det_2.5g-w600k_mbf": ["det_2.5g.onnx", "w600k_mbf.onnx"],
    "det_2.5g-w600k_r50": ["det_2.5g.onnx", "w600k_r50.onnx"],
    
    "det_10g-w600k_mbf": ["det_10g.onnx", "w600k_mbf.onnx"],
    "det_10g-w600k_r50": ["det_10g.onnx", "w600k_r50.onnx"],
}


ARCFACE_MODEL_NAMES = [
    name
    for name in ARCFACE_MODEL_WEIGHTS_NAMES.keys()
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ARCFACE_WEIGHTS_ROOT = ".arcface_weights"

ARCFACE_MODELS: dict[str, tuple[YakhyoArcFace, YakhyoSCRFD]] = {}


def _download_weights(weights_name, weights_path):
    assert weights_name in ARCFACE_WEIGHTS_URLS
    url = ARCFACE_WEIGHTS_URLS[weights_name]
    print(f"arcface: downloading {weights_name} from {url} and saved in {weights_path}")
    response = requests.get(url)
    response.raise_for_status()
    with open(str(weights_path), "wb") as file:
        file.write(response.content)
    pass


def _get_weights(model_name):
    assert model_name in ARCFACE_MODEL_NAMES

    # detector, recognizer
    scrfd_weights, af_weights = ARCFACE_MODEL_WEIGHTS_NAMES[model_name]

    weights_root = Path(ARCFACE_WEIGHTS_ROOT)
    weights_root.mkdir(parents=True, exist_ok=True)

    af_weights_path = weights_root / af_weights
    scrfd_weights_path = weights_root / scrfd_weights

    if not af_weights_path.exists():
        _download_weights(af_weights, af_weights_path)
    if not scrfd_weights_path.exists():
        _download_weights(scrfd_weights, scrfd_weights_path)
    
    assert af_weights_path.exists()
    assert scrfd_weights_path.exists()
    
    return af_weights_path, scrfd_weights_path
# end


def _get_model(model_name: str):
    global ARCFACE_MODELS
    if model_name in ARCFACE_MODELS:
        return ARCFACE_MODELS[model_name]
    
    af_weights_path, scrfd_weights_path = _get_weights(model_name)
    
    yarcface = YakhyoArcFace(str(af_weights_path))
    yscrfd = YakhyoSCRFD(str(scrfd_weights_path), input_size=(640,640))

    ARCFACE_MODELS[model_name] = (yarcface, yscrfd)
    return (yarcface, yscrfd)
# end


# ---------------------------------------------------------------------------
# ArcFace
# ---------------------------------------------------------------------------

class ArcFace:

    @staticmethod
    def represent(image: str | Path | np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = str(image)
            assert os.path.exists(filename)
            image: np.ndarray = cast(np.ndarray, cv2.imread(filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        recognizer, detector = _get_model(model_name)
        # YakhyoArcFace, YakhyoSCRFD

        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0: return None

        assert len(kpss) == 1
        embedding = recognizer.get_embedding(image, kpss[0])

        return embedding

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray) -> np.ndarray:
        emb = ArcFace.represent(image, self._model_name)
        assert isinstance(emb, np.ndarray)
        return emb

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        global ARCFACE_MODELS
        ARCFACE_MODELS.clear()
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------





