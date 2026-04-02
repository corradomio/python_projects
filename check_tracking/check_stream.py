import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics.engine.results import Results

from face_detectionx import FaceDetection
from sixdrepnetx import SixDRepNetMulti
from yolox import YOLOTracking, YOLOPose
from track_saver import TrackSaver
from stdlib.tprint import tprint


SKIP_FRAMES = 24*120


def main():
    # vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture(r"E:\Movies\Sokurov - 2002 - Arca Russa.mkv")

    tprint("Start stream ...")

    count = -1
    while True:
        count += 1
        rval, frame = vc.read()
        if not rval or frame is None:
            continue

        if count < SKIP_FRAMES:
            continue

        h, w, c = frame.shape
        frame = cv2.resize(frame, (w // 4, h // 4))
        annotated = frame.copy()

        # tprint(f"preview: {count}", force=False)

        cv2.imshow("preview", annotated)

        # ---------------------------------------------------------------

        key = cv2.waitKey(1)
        while key == 32:
            key = cv2.waitKey(500)
            key = 32 if key != 32 else 0
        if key == 27:  # exit on ESC
            break
    # end



if __name__ == "__main__":
    main()

