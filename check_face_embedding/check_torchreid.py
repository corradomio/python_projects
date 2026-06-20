from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from torchvision import transforms

from stdlib import csvx
from stdlib.is_instance import is_instance
from stdlib.tprint import tprint
from human.torchreid import TorchReID, TORCHREID_MODEL_NAMES


# IMAGE_DIRS = ["random_crop"]
# SCORE_DIR = "scores_maurizio_randomcrop"
IMAGE_DIRS = ["face"]
SCORE_DIR = "scores_maurizio_face"


ROOT_TRACKS = (
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-20\20260422_112233")
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")
    # Path(r".data_and_result\2026-05-19")
    Path(r".maurizio_dataset")
)

# ---------------------------------------------------------------------------

IMAGE_TYPE = np.ndarray
EMBEDDING = np.ndarray # [512]

# EMBEDDING_MODEL_NAMES = list(torchreid.models.__model_factory.keys())
EMBEDDING_MODEL_NAMES = TORCHREID_MODEL_NAMES

# EMBEDDING_MODEL_NAME = "osnet_x1_0"

# EMBEDDING_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EMBEDDING_MODEL_NAME: str = ""

EMBEDDING_CACHE = {}


def extract_feature(img_path: Path):
    global EMBEDDING_CACHE
    global EMBEDDING_LENGTH
    if img_path in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[img_path]

    feature: np.ndarray = TorchReID.represent(str(img_path), model_name=EMBEDDING_MODEL_NAME)
    assert is_instance(feature, np.ndarray)

    embedding = feature

    EMBEDDING_CACHE[img_path] = embedding
    return embedding


def load_image_files(dir: Path) -> list[Path]:
    assert is_instance(dir, Path)
    img_files = []
    for idir in IMAGE_DIRS:
        imgs_dir = dir / idir
        if not imgs_dir.exists(): continue
        for img_file in imgs_dir.glob("*.jpg"):
            # [h,w,c] uint8
            # image = cv2.imread(ifile)
            img_files.append(img_file)

    return img_files


def compute_embeddings(img_files: list[Path]) -> list[EMBEDDING]:
    embeddings = []
    for img_file in img_files:
        embeddings.append(extract_feature(img_file))
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
# end


def list_tracks(root_dir: Path):
    tracks = []
    for track,_,_ in root_dir.walk():
        if track.name.endswith("_DONE"):
            tracks.append(track)
    return tracks


def main():
    global EMBEDDING_MODEL_NAME
    global EMBEDDING_CACHE
    dirs = list_tracks(ROOT_TRACKS)
    n_dirs = len(dirs)

    for embedding_model_name in EMBEDDING_MODEL_NAMES:
        EMBEDDING_MODEL_NAME = embedding_model_name
        EMBEDDING_CACHE = {}

        scores = []
        score_file = Path(f"{SCORE_DIR}/torchreid-{EMBEDDING_MODEL_NAME}-scores.csv")
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

                similarity = embeddings_similarity(i_embeddings, j_embeddings)
                tprint(f"{i_dir} - {j_dir}: {similarity:.4} ({i+1:3}/{n_dirs})", force=False)

                if similarity > 0:
                    scores.append([i_dir.name, j_dir.name, similarity])
                pass
        # end
        if len(scores) > 0:
            csvx.dump(scores, str(score_file), header=["from", "to", "score"])
    # end
# end



if __name__ == "__main__":
    main()