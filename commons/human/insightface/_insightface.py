import os
from pathlib import Path
import logging

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSIGHTFACE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INSIGHTFACE_WEIGHTS_URLS = {

}

INSIGHTFACE_MODEL_WEIGHTS_NAMES = {

}


INSIGHTFACE_MODEL_NAMES = [
    # name.replace("/", "__")
    # for name in INSIGHTFACE_MODEL_WEIGHTS_NAMES.keys()
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc"
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

INSIGHTFACE_WEIGHTS_ROOT = ".insightface_weights"

INSIGHTFACE_MODELS = {}


def _get_model(model_name):
    if model_name in INSIGHTFACE_MODELS:
        return INSIGHTFACE_MODELS[model_name]

    assert model_name in INSIGHTFACE_MODEL_NAMES

    providers = ['CUDAExecutionProvider'] if INSIGHTFACE_DEVICE == "cuda" else ['CPUExecutionProvider']

    model = FaceAnalysis(
        name=model_name, root=INSIGHTFACE_WEIGHTS_ROOT,
        providers=providers  # Use 'CUDAExecutionProvider' for GPU
    )

    INSIGHTFACE_MODELS[model_name] = model
    return model
# end


# ---------------------------------------------------------------------------
# InsightFace
# ---------------------------------------------------------------------------

EMBEDDING_SIZE = 0


class InsightFace:

    @staticmethod
    def represent(image: str | Path | np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = str(image)
            assert os.path.exists(filename)
            image = cv2.imread(filename)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.open(filename).convert("RGB")
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        model = _get_model(model_name)

        global EMBEDDING_SIZE
        faces = model.get(image)
        if len(faces) == 0:
            embedding = np.zeros(EMBEDDING_SIZE, dtype=float)
        else:
            embedding = faces[0].embedding
            EMBEDDING_SIZE = embedding.shape
        # end
        return embedding
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return InsightFace.represent(image, self._model_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        INSIGHTFACE_MODELS.clear()
        pass
    # end