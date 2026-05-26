import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchreid
from scipy.spatial.distance import cdist
from stdlib.is_instance import is_instance
from stdlib import csvx
from stdlib.tprint import tprint

import numpy
import pyximport
pyximport.install(setup_args=dict(
    include_dirs=[numpy.get_include()]
))

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
except Exception as e:
    print(e)
    

IMAGE_TYPE = np.ndarray
EMBEDDING = np.ndarray # [512]

# EMBEDDING_MODEL_NAMES = list(torchreid.models.__model_factory.keys())
EMBEDDING_MODEL_NAMES = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512',
    'se_resnet50', 'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
    'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512',
    'inceptionresnetv2', 'inceptionv4', 'xception',
    'resnet50_ibn_a', 'resnet50_ibn_b', 'nasnsetmobile',
    'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet',
    'squeezenet1_0', 'squeezenet1_0_fc512', 'squeezenet1_1',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'mudeep', 'resnet50mid',
    # 'hacnn', # (h:160, w:64)
    'pcb_p6', 'pcb_p4', 'mlfn',
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
    'osnet_ibn_x1_0',
    'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25'
]

# EMBEDDING_MODEL_NAME = "osnet_x1_0"

EMBEDDING_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EMBEDDING_MODEL = torchreid.models.build_model(
#     name=EMBEDDING_MODEL_NAME,
#     num_classes=1000,
#     pretrained=True
# )
# EMBEDDING_MODEL.eval().to(EMBEDDING_DEVICE)
EMBEDDING_MODEL = None

FEATURE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


ROOT_TRACKS = (
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-20\20260422_112233")
    # Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-19")
    Path(r".data_and_result\2026-05-19")
)

IMAGE_DIRS = ["face", "random_crop"]

EMBEDDING_CACHE = {}


def extract_image_feature(img):
    img = FEATURE_TRANSFORMS(img).unsqueeze(0).to(EMBEDDING_DEVICE)
    with torch.no_grad():
        feat = EMBEDDING_MODEL(img)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()[0]


def extract_feature(img_path):
    if img_path in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[img_path]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = FEATURE_TRANSFORMS(img).unsqueeze(0).to(EMBEDDING_DEVICE)
    # with torch.no_grad():
    #     feat = EMBEDDING_MODEL(img)
    # feat = F.normalize(feat, p=2, dim=1)
    # return feat.cpu().numpy()[0]
    feature = extract_image_feature(img)
    EMBEDDING_CACHE[img_path] = feature
    return feature


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
        return 0

    dist_matrix = cdist(embeddings1, embeddings2, metric="cosine")
    similarity_matrix = 1 - dist_matrix
    max_similarity = similarity_matrix.max()
    return max_similarity
# end


def main():
    # global EMBEDDING_MODEL_NAME
    global EMBEDDING_MODEL
    global EMBEDDING_CACHE
    dirs = [dir.name for dir in ROOT_TRACKS.iterdir() if dir.name.endswith("_DONE")]
    n_dirs = len(dirs)

    for embedding_model_name in EMBEDDING_MODEL_NAMES:
        EMBEDDING_MODEL_NAME = embedding_model_name
        EMBEDDING_CACHE = {}

        EMBEDDING_MODEL = torchreid.models.build_model(
            name=EMBEDDING_MODEL_NAME,
            num_classes=1000,
            pretrained=True
        )
        EMBEDDING_MODEL.eval().to(EMBEDDING_DEVICE)

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

                similarity = embeddings_similarity(i_embeddings, j_embeddings)
                tprint(f"{i_dir} - {j_dir}: {similarity:.4} ({i+1:3}/{n_dirs})", force=False)

                scores.append([i_dir, j_dir, similarity])
                pass
        # end
        csvx.dump(scores, str(score_file), header=["from", "to", "score"])
    # end
# end



if __name__ == "__main__":
    main()