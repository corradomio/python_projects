from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from scipy.spatial.distance import cdist

from stdlib import csvx
from stdlib.is_instance import is_instance
from stdlib.tprint import tprint

# DeepFace.extract_faces()


FACE_MODELS = [
    "opencv",
        # "retinaface",   # very slow
    "mtcnn",
        # "ssd",          # strange: intercepts only SOME images
        # "dlib",         # problems compiling the package
        # "mediapipe",    # module "mediapipe" has no attribute "solutions"
        # "yolov8n",
    "yolov8m",
        # "yolov8l",
        # "yolov11n",
        # "yolov11s",
    "yolov11m",
        # "yolov11l",
        # "yolov12n",
        # "yolov12s",
    "yolov12m",
        # "yolov12l",
    "yunet",
    "fastmtcnn",
        # "centerface",   # UnboundLocalError: cannot access local variable 'boxes_np' where it is not associated with a value
        # "skip"
]

ROOT_TRACKS = (
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-20\20260422_112233")
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")
    Path(r".data_and_result\2026-05-19")
)

IMAGE_DIRS = ["random_crop", "face_recognition", "face_recognition_fallback", "not_dress_well", "not_glove_well",
              "unauthorised_access", "unauthorised_operation_A", "unauthorised_machine_touching_B"]

IMAGE_CACHE = {}


def load_image(img_path: Path) -> np.ndarray:
    if img_path in IMAGE_CACHE:
        return IMAGE_CACHE[img_path]

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMAGE_CACHE[img_path] = img
    return img


def load_image_files(dir: str) -> list[Path]:
    img_files = []
    for idir in IMAGE_DIRS:
        imgs_dir = ROOT_TRACKS / f"{dir}/{idir}"
        if not imgs_dir.exists(): continue
        for img_file in imgs_dir.glob("*.jpg"):
            if img_file.name.endswith("_whole.jpg"):
                continue
            # [h,w,c] uint8
            # image = cv2.imread(ifile)
            img_files.append(img_file)

    return img_files


def save_face(img_face, dir, image_name, FACE_MODEL):
    img_path: Path = ROOT_TRACKS / f"{dir}/face/{FACE_MODEL}/{image_name}"
    img_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(img_path), img_face)
    tprint(f"... {dir}/{image_name}", force=False)


def main():
    dirs: list[str] = [dir.name for dir in ROOT_TRACKS.iterdir() if dir.name.endswith("_DONE")]
    n_dirs = len(dirs)

    for FACE_MODEL in FACE_MODELS:
        tprint(f"-- {FACE_MODEL} --", force=True)
        for i in range(n_dirs - 1):
            i_dir = dirs[i]

            face_dir = ROOT_TRACKS/ f"{i_dir}/face/{FACE_MODEL}"
            if face_dir.exists(): continue

            i_images = load_image_files(i_dir)

            for i_image in i_images:
                img_array = load_image(i_image)
                try:
                    faces = DeepFace.extract_faces(
                        img_array,
                        detector_backend=FACE_MODEL,
                        enforce_detection=True,
                        normalize_face=False
                    )
                except FaceNotDetected:
                    faces = []
                if len(faces) == 0:
                    continue
                if len(faces) > 1:
                    tprint(f"{i_dir}/{i_image.name}: found {len(faces)} faces")

                # {
                #   "confidence": 0.97
                #   "face": np.ndarray[shape=(h,w,c), dtype=np.float64]
                #   "facial_area": {"left_eye": None, "right_eye": None, "h":52, "w": 53, "x":15", "y":266}
                # }
                img_face = faces[0]["face"]

                save_face(img_face, i_dir, i_image.name, FACE_MODEL)
                pass






if __name__ == "__main__":
    main()