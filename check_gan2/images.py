from PIL import Image
from path import Path as path
from numpy import asarray
import matplotlib.pyplot as plt

i = 0
for img_file in path("E:/Datasets/CelebA/gray_32").files("*.jpg"):
    img = Image.open(img_file)
    if i % 4000 == 0:
        plt.imshow(img, cmap="gray")
        plt.show()
    i += 1
    # break