import numpy as np
import pandas as pd
from path import Path as path
from random import random
import xml.etree.ElementTree as ET
from stdlib.tprint import tprint
from PIL import Image

# VOC_ROOT = r"E:/Datasets/VOC2012"
VOC_ROOT = r"D:/Datasets/VOC2012"

LABELS = path(f"{VOC_ROOT}/Annotations")
IMAGES = path(f"{VOC_ROOT}/JPEGImages")
LABELS_YOLO = path(f"{VOC_ROOT}/Annotations_yolo")
LABELS_VOC = path(f"{VOC_ROOT}/Annotations_voc")


SQUARED_IMAGES = path(f"{VOC_ROOT}/JPEGImages_square")



def main():
    tprint("Start", force=True)
    for ifile in IMAGES.files("*.jpg"):
        tprint(ifile)
        original_image = Image.open(ifile)
        s = max(*original_image.size)
        new_image = Image.new("RGB", (s, s), (255, 255, 255))
        new_image.paste(original_image, (0, 0))

        nfile = SQUARED_IMAGES / ifile.name
        new_image.save(nfile)

        pass
    tprint("Done", force=True)
    pass
# end


if __name__ == "__main__":
    main()
