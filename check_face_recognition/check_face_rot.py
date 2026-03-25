from typing import Iterator

import cv2 as cv
from ultralytics import YOLO
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from sixdrepnet import SixDRepNet
from ultralytics.engine.results import Results
from ultralytics.trackers import register_tracker

from sixdrepnet360 import SixDRepNet360
import face_detection
from stdlib.tprint import tprint

FONT_FACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 2
FONT_THICKNESS = 1
LINE_TYPE = cv.LINE_AA

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2

# Load an official or custom model

SIXDREPNET = SixDRepNet(dict_path="./6DRepNet_300W_LP_AFLW2000.pth")

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

N_FRAMES = 1

def main():
    # vc = cv.VideoCapture(r"E:\Movies\FILM - Polar - 2019.iTALiAN.WEBRiP.XviD-PRiME.avi")
    vc = cv.VideoCapture(0)

    count = -1
    rval = True
    while rval:
        count += 1
        rval, frame = vc.read()

        if not rval or frame is None:
            continue

        # ---

        pitch, yaw, roll = SIXDREPNET.predict(frame)
        SIXDREPNET.draw_axis(frame, pitch, yaw, roll)

        # ---

        cv.imshow("preview", frame)

        key = cv.waitKey(1)
        while key == 32:
            key = cv.waitKey(500)
            key = 32 if key != 32 else 0
        if key == 27:  # exit on ESC
            break
    # end
# end


if __name__ == "__main__":
    main()
