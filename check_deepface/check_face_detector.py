import datetime

import cv2
import os
from datetime import time
from path import Path
from deepface.models.Detector import Detector, FacialAreaRegion

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from deepface.models import face_detection
from pprint import pprint
from deepface import DeepFace

DF_BACKENDS = [
    "opencv",
    "ssd",
    # "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8n",
    # "yolov8m",
    # "yolov8l",
    "yolov11n",
    # "yolov11s",
    # "yolov11m",
    # "yolov11l",
    "yolov12n",
    # "yolov12s",
    # "yolov12m",
    # "yolov12l",
    "yunet",
    "centerface",
]


IMAGES_ROOT = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\tmp_saved")

def is_face_recognition(path: str) -> bool:
    return "face_recognition" in path and path.endswith(".jpg")

def main():
    for face_file in IMAGES_ROOT.walkfiles(is_face_recognition):
        print(face_file)
        face_image = cv2.imread(face_file)

        for backend in DF_BACKENDS:
            face_detector: Detector = DeepFace.build_model(backend, "face_detector")
            start = datetime.datetime.now()
            facial_regions: list[FacialAreaRegion] = face_detector.detect_faces(face_image)
            delta = int((datetime.datetime.now() - start).microseconds/1000)
            if len(facial_regions) == 0:
                # print("...", backend, ": no faces [", delta, "ms]")
                pass
            else:
                print("...", backend, ":", len(facial_regions), "faces [", delta, "ms]")
            pass


if __name__ == "__main__":
    main()
