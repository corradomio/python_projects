import cv2
from pathlib import Path
import numpy as np
import torch
from arcface import ArcFace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCFACE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCFACE_WEIGHTS_URLS = {

}

ARCFACE_MODEL_WEIGHTS_NAMES = {

}


ARCFACE_MODEL_NAMES = [
    # name.replace("/", "__")
    # for name in ARCFACE_MODEL_WEIGHTS_NAMES.keys()
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc"
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ARCFACE_WEIGHTS_ROOT = ".insightface_weights"

ARCFACE_MODELS = {}


def _get_model(model_name):
    if model_name in ARCFACE_MODELS:
        return ARCFACE_MODELS[model_name]

    assert model_name in ARCFACE_MODEL_NAMES

    model_path = _get_model_path(model_name)

    model = ArcFace(model_path)


    ARCFACE_MODELS[model_name] = model
    return model
# end


# ---------------------------------------------------------------------------
# InsightFaceReID
# ---------------------------------------------------------------------------

class ArcFace:

    @staticmethod
    def represent(image: str | Path | np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = image
            image = cv2.imread(filename)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.open(filename).convert("RGB")
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        model = _get_model(model_name)

        faces = model.get(image)
        assert len(faces) > 0
        return faces[0].embedding
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return ArcFace.represent(image, self._model_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        ARCFACE_MODELS.clear()
        pass
    # end
# end
