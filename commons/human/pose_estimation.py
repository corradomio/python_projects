import torch
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results, Keypoints

# List of YOLO models
# https://docs.ultralytics.com/models

# https://docs.ultralytics.com/tasks/pose#results-output
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml
#
#  0 Nose
#  1 Left Eye       occhio
#  2 Right Eye
#  3 Left Ear       orecchio
#  4 Right Ear
#  5 Left Shoulder  spalla
#  6 Right Shoulder
#  7 Left Elbow     gomito
#  8 Right Elbow
#  9 Left Wrist     polso
# 10 Right Wrist
# 11 Left Hip       anca
# 12 Right Hip
# 13 Left Knee      ginoccio
# 14 Right Knee
# 15 Left Ankle     caviglia
# 16 Right Ankle


# https://docs.ultralytics.com/tasks/pose#results-output
#
# Attribute	                Type	        Shape	    Description
# result.keypoints	        Keypoints	    (N)	        Keypoints.
# result.keypoints.data	    torch.float32	(N,K,2/3)	x,y plus optional visibility/confidence.
# result.keypoints.xy	    torch.float32	(N,K,2)	    Pixel keypoints.
# result.keypoints.xyn	    torch.float32	(N,K,2)	    Normalized keypoints.
# result.boxes	            Boxes	        (N)	        Instance boxes.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSE_MODELS = [
    "yolov8n-pose",
    "yolov8s-pose",
    "yolov8m-pose",
    "yolov8l-pose",
    "yolov8x-pose",
    "yolov8x-pose-p6",

    "yolo11n-pose",
    "yolo11s-pose",
    "yolo11m-pose",
    "yolo11l-pose",
    "yolo11x-pose",

    "YOLO26n-pose",
    "YOLO26s-pose",
    "YOLO26m-pose",
    "YOLO26l-pose",
    "YOLO26x-pose",
]

YOLO_WEIGHTS_ROOT = ".yolo_weights"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

YOLO_POSE_MODELS = {}

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
    if model_name in YOLO_POSE_MODELS:
        return YOLO_POSE_MODELS[model_name]

    weights_path = _get_weights(model_name)

    model = YOLO(str(weights_path)).eval().to(YOLO_DEVICE)
    YOLO_POSE_MODELS[model_name] = model
    return model


# ---------------------------------------------------------------------------
# HumanPose
# ---------------------------------------------------------------------------


class PoseKeypoints:
    def __init__(self, keypoints: Keypoints|None):

        # results: Results          (1)
        #   boxes: Boxes            (1,6)
        #   keypoints: Keypoints    (1,17,3)
        #       conf: Tensor[1,17]
        #       data: Tensor[1,17,3]    (x,y,confidence)
        #       xy: Tensor[1,17,2]
        #       xyn: Tensor[1,17,2
        #   masks: None
        #   names: dict[index, "object_type"]

        self.keypoints = keypoints

    def draw(self, image: np.ndarray, threshold:float = 0.5, thickness: int=2):
        assert isinstance(image, np.ndarray)

        keypoints_xy = self.keypoints.xy
        keypoints_conf = self.keypoints.conf

        if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
            return image

        # COCO 17-keypoint skeleton (edges between keypoints)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        for person_idx, (kpts, confs) in enumerate(zip(keypoints_xy, keypoints_conf)):
            kpts = kpts.cpu().numpy()  # Shape: (17, 2) [x, y]
            confs = confs.cpu().numpy()  # Shape: (17,) [confidence]

            # Draw keypoints
            for i, (x, y) in enumerate(kpts):
                if confs[i] > threshold:  # Draw keypoints with sufficient confidence
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw skeleton lines
            for (start, end) in skeleton:
                if confs[start] > threshold and confs[end] > threshold:
                    start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                    end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                    cv2.line(image, start_pt, end_pt, (255, 0, 0), thickness)
        pass
    # end

    def as_dict(self) -> dict:
        #  0 Nose
        #  1 Left Eye       occhio
        #  2 Right Eye
        #  3 Left Ear       orecchio
        #  4 Right Ear
        #  5 Left Shoulder  spalla
        #  6 Right Shoulder
        #  7 Left Elbow     gomito
        #  8 Right Elbow
        #  9 Left Wrist     polso
        # 10 Right Wrist
        # 11 Left Hip       anca
        # 12 Right Hip
        # 13 Left Knee      ginoccio
        # 14 Right Knee
        # 15 Left Ankle     caviglia
        # 16 Right Ankle
        keypoints = self.keypoints
        if len(keypoints) == 0:
            return {}

        kp = {
            "nose": keypoints.data[0, 0].cpu().numpy().tolist(),
            "left": {
                "eye":      keypoints.data[0, 1].cpu().numpy().tolist(),
                "ear":      keypoints.data[0, 3].cpu().numpy().tolist(),
                "shoulder": keypoints.data[0, 5].cpu().numpy().tolist(),
                "elbow":    keypoints.data[0, 7].cpu().numpy().tolist(),
                "wrist":    keypoints.data[0, 9].cpu().numpy().tolist(),
                "hip":      keypoints.data[0, 11].cpu().numpy().tolist(),
                "knee":     keypoints.data[0, 13].cpu().numpy().tolist(),
                "ankle":    keypoints.data[0, 15].cpu().numpy().tolist(),
            },
            "right": {
                "eye":      keypoints.data[0, 2].cpu().numpy().tolist(),
                "ear":      keypoints.data[0, 4].cpu().numpy().tolist(),
                "shoulder": keypoints.data[0, 6].cpu().numpy().tolist(),
                "elbow":    keypoints.data[0, 8].cpu().numpy().tolist(),
                "wrist":    keypoints.data[0, 10].cpu().numpy().tolist(),
                "hip":      keypoints.data[0, 12].cpu().numpy().tolist(),
                "knee":     keypoints.data[0, 14].cpu().numpy().tolist(),
                "ankle":    keypoints.data[0, 16].cpu().numpy().tolist(),
            }
        }
        return kp
# end

class HumanPose:

    @staticmethod
    def pose(image: str|Path|np.ndarray, model_name: str, params: dict=None) -> PoseKeypoints:
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

        pose_model = _get_model(model_name)

        # batch=len(crops_need_to_run_yolo_pose),
        #                     conf=conf_person_thr,
        #                     classes=[0],  # person,
        #                     iou=0.5,
        #                     verbose=False,
        #                     device="cuda"

        human_poses: list[Results] = pose_model(
            image,
            classes=[0],
            iou=params.get("iou", 0.5),
            conf=params.get("conf", 0.7),
            device=YOLO_DEVICE
        )
        if len(human_poses) == 0:
            return PoseKeypoints(None)
        if len(human_poses) > 1:
            human_poses = human_poses[0:1]

        results: Results = human_poses[0]
        return PoseKeypoints(results.keypoints)

    # -----------------------------------------------------------------------

    def __init__(self, model_name: str, params: dict=None):
        self.model_name = model_name
        self.params = params

    def pose_estimation(self, image: str|Path|np.ndarray) -> PoseKeypoints:
        return HumanPose.pose(image, self.model_name, self.params)
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
