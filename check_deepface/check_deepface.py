import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from pprint import pprint
from deepface import DeepFace

DF_MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    # "DeepFace", # to use  tf 2.12   current 2.21
    "DeepID",
    "ArcFace",
    # "Dlib",     # https://dlib.net/
    "SFace",
    "GhostFaceNet",
    "Buffalo_L",
]

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

DF_MODELS = ["Facenet512"]
DF_BACKENDS = ["opencv"]


for model in DF_MODELS:
    for backend in DF_BACKENDS:

        print("---", backend, "/", model, "---")
        result: dict = DeepFace.verify(
            img1_path = "dataset/img1.jpg", img2_path = "dataset/img2.jpg",
            model_name=model, detector_backend=backend
        )
        print("   ", backend, "/", model, ": ", end="")
        pprint(result)


# dfs: list[pd.DataFrame] =DeepFace.find(img_path="dataset/img1.jpg", db_path="dataset")
# for df in dfs:
#     df.head(100)


# objs: list[dict] = DeepFace.analyze(
#   img_path = "dataset/img4.jpg", actions = ["age", "gender", "race", "emotion"]
# )
#
# pprint(objs)
