from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import torch
import torchvision.transforms.v2
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_keypoints
from ultralytics import YOLO


POSE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# models
# https://platform.ultralytics.com/ultralytics/yolo26
# YOLO26n|s|m|l|x-pose
#
# https://platform.ultralytics.com/ultralytics/yolo11
# YOLO11n|s|m|l|x-pose


def _yolo11_pose_factory():
    YOLO_POSE_MODEL = YOLO("yolo11n-pose.pt")
    YOLO_POSE_MODEL.eval().to(POSE_DEVICE)
    return YOLO_POSE_MODEL

def _tvision_pose_factory():
    TVISION_POSE_MODEL = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    TVISION_POSE_MODEL.eval().to(POSE_DEVICE)
    return TVISION_POSE_MODEL



POSE_MODEL_FACTORY = {
    "yolo11n-pose": _yolo11_pose_factory,
    "tvision-pose": _tvision_pose_factory
}


POSE_MODEL_NAMES = list(POSE_MODEL_FACTORY.keys())


POSE_MODELS = { }



transform = transforms.Compose([
    transforms.v2.ToImage(),
    transforms.v2.ToDtype(torch.float32, scale=True)
])

ToImage = transforms.v2.ToImage()
ToPILImage = transforms.ToPILImage()



def _get_pose_model(model_name: str):
    if model_name in POSE_MODELS:
        return POSE_MODELS[model_name]

    model_factory = POSE_MODEL_FACTORY[model_name]
    model = model_factory()
    POSE_MODELS[model_name] = model
    return model


class HumanPose:

    @staticmethod
    def pose(image: str|Path|np.ndarray, bbox, model_name: str):
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = cv2.imread(image_path)

        model = _get_pose_model(model_name)






