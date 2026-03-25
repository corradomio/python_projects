import face_detection
from face_detection.torch_utils import get_device
import cv2 as cv
import numpy as np

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2


class FaceDetection:

    def __init__(
            self,
            name: str = "DSFDDetector",
            confidence_threshold: float = 0.5,
            nms_iou_threshold: float = 0.3,
            device=get_device(),
            max_resolution: int = None,
            fp16_inference: bool = False,
            clip_boxes: bool = False
    ):
        self._model = face_detection.build_detector(
            name=name,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            max_resolution=max_resolution,
            fp16_inference=fp16_inference,
            clip_boxes=clip_boxes
        )


    def detect(self, image, shrink=1.0) -> np.ndarray:
            return self._model.detect(image, shrink=shrink)

    def plot(self, frame, boxes):
        def _box(box):
            tx, ty, bx, by, prob = box
            top_left_corner = (int(tx), int(ty))
            bottom_right_corner = (int(bx), int(by))
            cv.rectangle(frame, top_left_corner, bottom_right_corner, LINE_COLOR, LINE_THICKNESS)

        for box in boxes:
            _box(box)
    # end
# end
