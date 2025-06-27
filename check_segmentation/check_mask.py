from path import Path as path
from stdlib.tprint import tprint
import imageio.v3 as iio
from mask_classes import mask_classes
import numpy as np

# IMAGES_DIR = r"D:\Projects.github\article_projects\article_waterdrops\images"
IMAGES_DIR = r"images"



def main():
    images_dir = path(IMAGES_DIR)
    for dir in images_dir.dirs():
        tprint(dir, force=True)
        for drop_mask in dir.files("drop_mask-*.png"):
            # train_segmentation(scene_file)
            image: np.ndarray = iio.imread(drop_mask)
            imclases = mask_classes(image)
            print(image.dtype)



if __name__ == "__main__":
    main()
