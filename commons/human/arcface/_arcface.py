import os
import cv2
from pathlib import Path
import numpy as np
import torch
import requests
from torch.nn import DataParallel

from . import resnet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCFACE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCFACE_WEIGHTS_URLS = {
    'resnet_face18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

ARCFACE_MODEL_WEIGHTS_NAMES = {
    'resnet_face18': 'resnet18-5c106cde.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'models/resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}

ARCFACE_MODEL_CLASSES = {
    'resnet_face18': resnet.resnet_face18,
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    # 'resnet_face18': resnet.resnet_face18,
}



ARCFACE_MODEL_NAMES = [
    name.replace("/", "__")
    for name in ARCFACE_MODEL_WEIGHTS_NAMES.keys()
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ARCFACE_WEIGHTS_ROOT = ".arcface_weights"

ARCFACE_MODELS = {}


def _load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, weights_only=False)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def _download_weights(model_name, weights_path):
    url = ARCFACE_WEIGHTS_URLS[model_name]
    print(f"arcface: downloading {model_name} from {url} and saved in {weights_path}")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(weights_path, "wb") as file:
        file.write(response.content)
    pass


def _get_weights(model_name: str) -> str:
    weights_name = ARCFACE_MODEL_WEIGHTS_NAMES[model_name]
    weights_path = Path(ARCFACE_WEIGHTS_ROOT) / weights_name

    if not weights_path.exists():
        _download_weights(model_name, weights_path)

    assert weights_path.exists()
    return str(weights_path)


def _get_model(model_name):
    if model_name in ARCFACE_MODELS:
        return ARCFACE_MODELS[model_name]

    assert model_name in ARCFACE_MODEL_NAMES

    if model_name == "resnet_face18":
        model = ARCFACE_MODEL_CLASSES[model_name](False)
    else:
        model = ARCFACE_MODEL_CLASSES[model_name](True)

    weights_path = _get_weights(model_name)

    model = _load_model(model, weights_path)

    # model.load_state_dict(torch.load(weights_path, weights_only=False))
    # model.eval().to(ARCFACE_DEVICE)

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
            filename = str(image)
            assert os.path.exists(filename)
            image = cv2.imread(filename, 0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.open(filename).convert("RGB")
        elif isinstance(image, np.ndarray):
            # array = cast(np.ndarray, image)
            # image = Image.fromarray(array, mode="RGB")
            pass

        image = np.dstack((image, np.fliplr(image)))
        image = image.transpose((2, 0, 1))
        image = image[:, np.newaxis, :, :]
        image = image.astype(np.float32, copy=False)
        image -= 127.5
        image /= 127.5

        model = _get_model(model_name)

        data = torch.from_numpy(image.reshape((1, -1))).to(ARCFACE_DEVICE)
        output = model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))

        return feature
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
