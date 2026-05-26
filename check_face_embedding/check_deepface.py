from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from scipy.spatial.distance import cdist

from stdlib import csvx
from stdlib.is_instance import is_instance
from stdlib.tprint import tprint

# DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")

IMAGE_TYPE = np.ndarray
EMBEDDING = np.ndarray # [512]

EMBEDDING_LENGTH = 0

# 'VGG-Face' = {ABCMeta} <class 'deepface.models.facial_recognition.VGGFace.VggFaceClient'>
# 'OpenFace' = {ABCMeta} <class 'deepface.models.facial_recognition.OpenFace.OpenFaceClient'>
# 'Facenet' = {ABCMeta} <class 'deepface.models.facial_recognition.Facenet.FaceNet128dClient'>
# 'Facenet512' = {ABCMeta} <class 'deepface.models.facial_recognition.Facenet.FaceNet512dClient'>
# 'DeepFace' = {ABCMeta} <class 'deepface.models.facial_recognition.FbDeepFace.DeepFaceClient'>
# 'DeepID' = {ABCMeta} <class 'deepface.models.facial_recognition.DeepID.DeepIdClient'>
# 'Dlib' = {ABCMeta} <class 'deepface.models.facial_recognition.Dlib.DlibClient'>
# 'ArcFace' = {ABCMeta} <class 'deepface.models.facial_recognition.ArcFace.ArcFaceClient'>
# 'SFace' = {ABCMeta} <class 'deepface.models.facial_recognition.SFace.SFaceClient'>
# 'GhostFaceNet' = {ABCMeta} <class 'deepface.models.facial_recognition.GhostFaceNet.GhostFaceNetClient'>
# 'Buffalo_L' = {ABCMeta} <class 'deepface.models.facial_recognition.Buffalo_L.Buffalo_L'>

EMBEDDING_MODEL_NAMES = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "Dlib", "ArcFace", "SFace", "GhostFaceNet",
    "Buffalo_L"
]
EMBEDDING_MODEL_NAME = None

ROOT_TRACKS = (
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-20\20260422_112233")
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")
    Path(r".data_and_result\2026-05-19")
)

IMAGE_DIRS = ["face", "random_crop"]

EMBEDDING_CACHE = {}

IMAGE_CACHE = {}


def load_image(img_path):
    if img_path in IMAGE_CACHE:
        return IMAGE_CACHE[img_path]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMAGE_CACHE[img_path] = img
    return img


def extract_feature(img_path):
    global EMBEDDING_CACHE
    global EMBEDDING_LENGTH
    if img_path in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[img_path]

    feature = DeepFace.represent(str(img_path), model_name=EMBEDDING_MODEL_NAME, detector_backend="skip")
    # {
    #   "embedding": [...]
    #   'face_confidence': 0.92,
    #   'facial_area': {
    #       'h': 45,
    #       'left_eye': None,
    #       'right_eye': None,
    #       'w': 45,
    #       'x': 4,
    #       'y': 11
    #    }
    # }
    if len(feature) == 1:
        embedding = feature[0]["embedding"]
        EMBEDDING_LENGTH = len(embedding)
    elif len(feature) > 1:
        embedding = feature[0]["embedding"]
        EMBEDDING_LENGTH = len(embedding)
    else:
        embedding = [0.]*EMBEDDING_LENGTH
    embedding = np.array(embedding)

    EMBEDDING_CACHE[img_path] = embedding
    return embedding


def load_image_files(dir: str) -> list[Path]:
    img_files = []
    for idir in IMAGE_DIRS:
        imgs_dir = ROOT_TRACKS / f"{dir}/{idir}"
        if not imgs_dir.exists(): continue
        for img_file in imgs_dir.glob("*.jpg"):
            # [h,w,c] uint8
            # image = cv2.imread(ifile)
            img_files.append(img_file)

    return img_files



def compute_embeddings(img_files: list[Path]) -> list[EMBEDDING]:
    embeddings = []
    for img_file in img_files:
        try:
            embeddings.append(extract_feature(img_file))
        except FaceNotDetected as e:
            pass
        except Exception as e:
            pass
    return embeddings


def embeddings_similarity(embeddings1: list[EMBEDDING], embeddings2: list[EMBEDDING]) -> float:
    # WARNING: similarity is in range [0,1]
    #   WITH 1 THE BEST  SIMILARITY
    #   WITH 0 THE WORST SIMILARITY
    assert is_instance(embeddings1, list[EMBEDDING])
    assert is_instance(embeddings2, list[EMBEDDING])

    ##It just uses them with scipy.spatial.distance.cdist and cosine distance for similarity.
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return 0.

    dist_matrix = cdist(embeddings1, embeddings2, metric="cosine")
    similarity_matrix = 1 - dist_matrix
    max_similarity = similarity_matrix.max()
    return max_similarity



def main():
    global EMBEDDING_MODEL_NAME
    global EMBEDDING_CACHE
    dirs = [dir.name for dir in ROOT_TRACKS.iterdir() if dir.name.endswith("_DONE")]
    n_dirs = len(dirs)

    for embedding_model_name in EMBEDDING_MODEL_NAMES:
        EMBEDDING_MODEL_NAME = embedding_model_name
        EMBEDDING_CACHE = {}

        scores = []
        score_file = Path(f"scores/{EMBEDDING_MODEL_NAME}-scores.csv")
        if score_file.exists(): continue

        tprint(f"-- {EMBEDDING_MODEL_NAME} --", force=True)

        for i in range(n_dirs-1):
            i_dir = dirs[i]
            i_images = load_image_files(i_dir)
            i_embeddings = compute_embeddings(i_images)
            for j in range(i+1, n_dirs):

                j_dir = dirs[j]
                j_images = load_image_files(j_dir)
                j_embeddings = compute_embeddings(j_images)

                similarity = embeddings_similarity(i_embeddings, j_embeddings) + 0.
                tprint(f"{i_dir} - {j_dir}: {similarity:.4} ({i+1:3}/{n_dirs})", force=False)

                scores.append([i_dir, j_dir, similarity])
                pass
        # end
        csvx.dump(scores, str(score_file), header=["from", "to", "score"])
# end
# end



if __name__ == "__main__":
    main()