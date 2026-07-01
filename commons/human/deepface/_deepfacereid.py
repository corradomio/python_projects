import os

from deepface import DeepFace
import numpy as np
from pathlib import Path

DEEPFACE_MODEL_NAMES = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",     # it requires tf-keras <= 2.12 BUT it is installed 2.21
    "DeepID",
    # "Dlib",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
    "Buffalo_L"
]


class DeepFaceReID:

    @staticmethod
    def represent(image: str|Path|np.ndarray, model_name: str):
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, Path):
            filename = str(image)
            assert os.path.exists(filename)
            image = filename
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        return DeepFace.represent(image, model_name, detector_backend="skip")
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return DeepFaceReID.represent(image, self._model_name)


    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
       pass
# end


