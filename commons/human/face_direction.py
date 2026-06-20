import gdown
import requests
import cv2
from pathlib import Path
import numpy as np
from sixdrepnet import SixDRepNet
from .sixdrepnet360 import SixDRepNet360

# https://github.com/thohemp/6DRepNet
# https://github.com/thohemp/6DRepNet360

# https://github.com/Shohruh72/SixDRepNet   (other models)


FACE_DIRECTION_MODEL_NAMES = [
    "6DRepNet",
    "6DRepNet360",
    "6DRepNet360_Full_Rotation"
]


FACE_DIRECTION_WEIGHT_URLS = {
    "6DRepNet":"https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_?usp=sharing",  # (Google)
    "6DRepNet360":"https://cloud.ovgu.de/s/wWCitDxp9t79xkP/download/6DRepNet360_300W_LP.pth",
    "6DRepNet360_Full_Rotation":"https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
}


FACE_DIRECTION_WEIGHT_NAMES = {
    "6DRepNet": "6DRepNet360_300W_LP.pth",
    "6DRepNet360": "6DRepNet_300W_LP_AFLW2000.pth",
    "6DRepNet360_Full_Rotation": "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
}


FACE_DIRECTION_WEIGHTS_ROOT = ".6drepnet_weights"

FACE_DIRECTION_MODELS = {}


def _download_weights(model_name: str, weights_path: Path):
    assert model_name in FACE_DIRECTION_WEIGHT_URLS

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    url = FACE_DIRECTION_WEIGHT_URLS[model_name]
    print(f"face_direction: downloading {model_name} from {url} and saved in {weights_path}")

    if model_name == "6DRepNet":
        gdown.download(url, str(weights_path), quiet=True)
    else:
        response = requests.get(url)
        response.raise_for_status()
        with open(weights_path, "wb") as file:
            file.write(response.content)
        pass
# end


def _get_weights(model_name: str) -> str:
    weights_file = FACE_DIRECTION_WEIGHT_NAMES[model_name]
    weights_path = Path(FACE_DIRECTION_WEIGHTS_ROOT) / weights_file
    if not weights_path.exists():
        _download_weights(model_name, weights_path)
    return str(weights_path)


def _get_model(model_name: str):
    if model_name in FACE_DIRECTION_MODELS:
        return FACE_DIRECTION_MODELS[model_name]

    assert model_name in FACE_DIRECTION_MODEL_NAMES

    weights_path = _get_weights(model_name)

    if "360" in model_name:
        model = SixDRepNet360(dict_path=weights_path)
    else:
        model = SixDRepNet(dict_path=weights_path)

    FACE_DIRECTION_MODELS[model_name] = model
    return model



class FaceDirection:

    @staticmethod
    def direction(image: str|Path|np.ndarray, model_name: str="6DRepNet") -> tuple[float, float, float]:
        # return pitch, yaw, roll in DEGREES
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        model = _get_model(model_name)

        pitch, yaw, roll = model.predict(image)
        return pitch, yaw, roll

    def __init__(self, model_name: str="6DRepNet"):
        self.model_name = model_name

    def detect(self, image: str|Path|np.ndarray):
        return FaceDirection.direction(image, model_name=self.model_name)



