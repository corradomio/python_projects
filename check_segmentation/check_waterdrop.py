import imageio.v3 as iio
import numpy as np
from path import Path as path
from stdlib.tprint import tprint
from joblib import Parallel, delayed


IMAGES_DIR = r"D:\Projects.github\article_projects\article_waterdrops\images"


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


def train_segmentation(scene_file: path):
    tprint(f"... {scene_file.stem}")

    dir: path = scene_file.parent

    scene_id = idof(scene_file.stem)
    mask_file = dir / f"drop_mask-{scene_id}.png"

    scene_image: np.ndarray = iio.imread(scene_file)
    mask_image: np.ndarray = iio.imread(mask_file)
    pass
# end


def train_nn():
    images_dir = path(IMAGES_DIR)
    for dir in images_dir.dirs():
        tprint(dir, force=True)
        for scene_file in dir.files("drop_scene-*.png"):
            train_segmentation(scene_file)
        # end
    # end
# end


def main():
    train_nn()
# end


if __name__ == "__main__":
    main()
