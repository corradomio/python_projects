
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.trackers import register_tracker


class YOLOPose(YOLO):
    def __init__(self, model: str | Path = "yolo26n-pose.pt", task: str | None = None, verbose: bool = False):
        super().__init__(model, task, verbose)  # Load an official Detect model


class YOLOTracking(YOLO):
    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        super().__init__(model, task, verbose)  # Load an official Detect model
        register_tracker(self, False)

    def predict(
            self,
            source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
            stream: bool = False,
            predictor=None,
            **kwargs: Any,
    ) -> list[Results]:
        # list[Results]
        # Results:
        return super().predict(source, stream, predictor, **kwargs)

    def plot(self, annotated, results: list[Results]):
        frame = annotated.copy()
        for r in results:
            frame = r.plot(img=frame)
        annotated[:, :, :] = frame

    def track_boxes(self, results: list[Results], threshold: float=0.5):
        # creates a dictionary 'track_id: bounding_box'
        # to use with 'face_recognition' and to associate the face with the track
        # Results:
        track_boxes = {}

        for r in results:
            n_boxes = r.boxes.shape[0]
            for b in range(n_boxes):
                box = r.boxes[b]
                # cls
                # conf
                # id:
                # is_track
                # xywh(n)
                # xyxy(n)
                if not box.is_track: continue
                if box.cls != 0: continue
                if box.conf < threshold: continue
                track_id = int(box.id)
                xyxy = box.xyxy.cpu().detach().numpy()

                track_boxes[track_id] = xyxy
        # end
        return track_boxes

# end