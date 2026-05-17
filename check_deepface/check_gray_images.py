import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace


TMP_RESULT = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\tmp_result")


def load_images(folder: Path) -> list[np.ndarray]:
    images = []
    for img_path in folder.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        images.append(img)
    return images

def compute_embeddings(images: list[np.ndarray]) -> np.ndarray:
    embeddings = []
    for img in images:
        emb = DeepFace.represent(img)
        embeddings.append(emb)


def main():

    # rc_0_0_DONE = TMP_RESULT / "0_0_DONE/random_crop"
    # rc_0_1_DONE = TMP_RESULT / "0_1_DONE/random_crop"
    #
    # images_0_0 = load_images(rc_0_0_DONE)
    # images_0_1 = load_images(rc_0_1_DONE)
    #
    # embeddings_0_0 = compute_embeddings(images_0_0)
    # embeddings_0_1 = compute_embeddings(images_0_1)

    for done in TMP_RESULT.iterdir():
        for folder in done.iterdir():
            if folder.is_file(): continue
            for img_path in folder.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                

    pass




    pass


if __name__ == "__main__":
    main()
