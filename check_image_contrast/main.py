import numpy as np
from path import Path as path
import cv2 as cv
import matplotlib.pyplot as plt


print(cv.__version__)


def analyze_image_bw(img_path: path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    hist0 = cv.calcHist([img], [0], None, [256], [0, 256]).astype(np.uint32)
    # if max(hist0) <= 35000:
    #     return

    histf = hist0[hist0>5000]
    if len(histf>100):
        return


    print(img_path)
    plt.plot(hist0, c='b')
    plt.title("gray: " + img_path.stem)
    plt.show()

    pass


def analyze_image_color(img_path: path):
    print(img_path)
    img = cv.imread(img_path, cv.IMREAD_COLOR_RGB)
    hist0 = cv.calcHist([img], [0], None, [256], [0, 256]).astype(np.uint32)
    hist1 = cv.calcHist([img], [1], None, [256], [0, 256]).astype(np.uint32)
    hist2 = cv.calcHist([img], [2], None, [256], [0, 256]).astype(np.uint32)

    plt.plot(hist0, c='r')
    plt.plot(hist1, c='g')
    plt.plot(hist2, c='b')
    plt.title("color: " + img_path.stem)
    plt.show()

    pass


# IMAGES = "2026-02-26"
IMAGES = "2026-02-26_saved"


def main():
    root = path(IMAGES)
    for f in root.walkfiles("*.jpg"):
        # analyze_image_color(f)
        analyze_image_bw(f)
        # break
    pass


if __name__ == "__main__":
    main()
