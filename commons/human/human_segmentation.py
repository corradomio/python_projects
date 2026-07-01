import torch
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results, Keypoints


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENTATION_MODELS = [
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8l-seg",
    "yolov8x-seg",

    "yolo11n-seg",
    "yolo11s-seg",
    "yolo11m-seg",
    "yolo11l-seg",
    "yolo11x-seg",

    "YOLO26n-seg",
    "YOLO26s-seg",
    "YOLO26m-seg",
    "YOLO26l-seg",
    "YOLO26x-seg",
]

YOLO_WEIGHTS_ROOT = ".yolo_weights"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

YOLO_SEG_MODELS: dict[str, YOLO] = {}

YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _download_weights(model_name: str, weights_path) -> None:
    yolo_name = f"{model_name}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    YOLO(yolo_name).save(str(weights_path))
    pass


def _get_weights(model_name: str) -> Path:
    weights_path = Path(YOLO_WEIGHTS_ROOT) / f"{model_name}.pt"
    if not weights_path.exists():
        _download_weights(model_name, weights_path)
    return weights_path


def _get_model(model_name: str) -> YOLO:
    if model_name in YOLO_SEG_MODELS:
        return YOLO_SEG_MODELS[model_name]

    weights_path = _get_weights(model_name)

    model = YOLO(str(weights_path)).eval().to(YOLO_DEVICE)
    YOLO_SEG_MODELS[model_name] = model
    return model


class HumanSegmentation:

    @staticmethod
    def identify_humans(image: str | Path | np.ndarray, model_name: str, params: dict=None):
        if params is None:
            params = {}

        if isinstance(image, (str, Path)):
            image_path = str(image)
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image, np.ndarray):
            data = image
            image = Image.fromarray(data, mode="RGB")
        else:
            raise ValueError(f"Unsupported image {type(image)}")

        assert isinstance(image, Image.Image)
        assert model_name in SEGMENTATION_MODELS, f"Model {model_name} not existent"

        seg_model = _get_model(model_name)

        human_seg: list[Results] = seg_model(image)

        human_mask = human_seg[0].masks.data[0]
        pass

    # -----------------------------------------------------------------------

    @staticmethod
    def dispose():
        global YOLO_SEG_MODELS
        keys = list(YOLO_SEG_MODELS.keys())
        for k in keys:
            YOLO_SEG_MODELS[k].to("cpu")
            del YOLO_SEG_MODELS[k]
    # end
# end