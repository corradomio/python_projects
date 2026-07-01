import cv2
from pathlib import Path
import numpy as np
import torch
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSIGHTFACEREID_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INSIGHTFACEREID_WEIGHTS_URLS = {

}

INSIGHTFACEREID_MODEL_WEIGHTS_NAMES = {

}


INSIGHTFACEREID_MODEL_NAMES = [
    # name.replace("/", "__")
    # for name in INSIGHTFACEREID_MODEL_WEIGHTS_NAMES.keys()
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc"
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

INSIGHTFACEREID_WEIGHTS_ROOT = ".insightface_weights"

INSIGHTFACEREID_MODELS = {}


def _get_model(model_name):
    if model_name in INSIGHTFACEREID_MODELS:
        return INSIGHTFACEREID_MODELS[model_name]

    assert model_name in INSIGHTFACEREID_MODEL_NAMES

    providers = ['CUDAExecutionProvider'] if INSIGHTFACEREID_DEVICE == "cuda" else ['CPUExecutionProvider']

    model = FaceAnalysis(
        name=model_name, root=INSIGHTFACEREID_WEIGHTS_ROOT,
        providers=providers  # Use 'CUDAExecutionProvider' for GPU
    )

    INSIGHTFACEREID_MODELS[model_name] = model
    return model
# end


# ---------------------------------------------------------------------------
# InsightFaceReID
# ---------------------------------------------------------------------------

class InsightFaceReID:

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
        return InsightFaceReID.represent(image, self._model_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        INSIGHTFACEREID_MODELS.clear()
        pass
    # end