import pandas as pd
import numpy as np
import pandas as pd
import skimage as skm
from joblib import Parallel, delayed
from path import Path as path
from skimage.color import rgb2gray
from skimage.transform import resize

import mnistio as mio
from timing import tprint

DATA_DIR="E:\Datasets\CartoonSet\cartoonset100k"

SIZE = 32


def load_image(f, count, asrgb=False):
    tprint(f"[{count+1:6}] Loading {f.stem} ...")
    # img = iio.read(f)
    img = skm.io.imread(f)
    img = skm.util.crop(img, ((50, 50), (50, 50), (0, 0)))
    if asrgb:
        img = img[:, :, 0:3]
    else:
        img = rgb2gray(img[:, :, 0:3])
    img = resize(img, (SIZE, SIZE))
    img = (img*255).astype(np.uint8)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    # plt.imshow(img[:,:,0])
    # plt.show()
    # plt.imshow(img[:,:,1])
    # plt.show()
    # plt.imshow(img[:,:,2])
    # plt.show()
    # plt.imshow(img[:,:,3])
    # plt.show()
    #
    # plt.imshow(img[:,:,0:3])
    # plt.show()
    # plt.imshow(img[:,:,:])
    # plt.show()
    return img
    pass


LABELS = [
    "eye_angle",
    "eye_lashes",
    "eye_lid",
    "chin_length",
    "eyebrow_weight",
    "eyebrow_shape",
    "eyebrow_thickness",
    "face_shape",
    "facial_hair",
    "hair",
    "eye_color",
    "face_color",
    "hair_color",
    "glasses",
    "glasses_color",
    "eye_slant",
    "eyebrow_width",
    "eye_eyebrow_distance"
]


def init_labels():
    return {
        "eye_angle": [],
        "eye_lashes": [],
        "eye_lid": [],
        "chin_length": [],
        "eyebrow_weight": [],
        "eyebrow_shape": [],
        "eyebrow_thickness": [],
        "face_shape": [],
        "facial_hair": [],
        "hair": [],
        "eye_color": [],
        "face_color": [],
        "hair_color": [],
        "glasses": [],
        "glasses_color": [],
        "eye_slant": [],
        "eyebrow_width": [],
        "eye_eyebrow_distance": []
    }


def load_labels(f: path):
    meta = {}
    f = f.replace(".png", ".csv")
    with open(f, mode="r") as fin:
        for l in fin:
            parts = l.split(",")
            parts[0] = parts[0].strip()[1:-1]
            parts[1] = int(parts[1].strip())
            parts[2] = int(parts[2].strip())
            meta[parts[0]] = [parts[1], parts[2]]
    # return meta
    return [meta[l][0] for l in LABELS]


def merge_labels(mnist_labels, labels):
    for k in labels:
        mnist_labels[k].append(labels[k][0])
    return mnist_labels


def load_cartoonset10k(asrgb=False):
    DATA_DIR = "E:\Datasets\CartoonSet\cartoonset10k"
    mnist_images = []
    mnist_labels = []
    data_dir = path(DATA_DIR)
    for f in data_dir.files("*.png"):
        # img = load_image(f, len(mnist_images), asrgb=asrgb)
        # mnist_images.append(img)
        labels = load_labels(f)
        mnist_labels.append(labels)
    # end
    # mnist = np.array(mnist_images)
    # mio.save_images(f"cartoonset10k{'-rgb' if asrgb else ''}-{SIZE}.gz", mnist, asrgb=asrgb, clast=True)
    labels = np.array(mnist_labels, dtype=np.uint8).reshape(-1)
    mio.save_labels(f"cartoonset10k-labels.gz", labels)
    # df = pd.DataFrame(data=mnist_labels)
    # df.to_csv("cartoonset10k.csv", index=False)
    pass


def load_cartoonset100k_dir(sdir, asrgb=False):
    mnist_images = []
    mnist_labels = []
    for f in sdir.files("*.png"):
        # img = load_image(f, len(mnist_images), asrgb=asrgb)
        # mnist_images.append(img)
        labels = load_labels(f)
        mnist_labels.append(labels)
    # end
    # mnist = np.array(mnist_images)
    # mio.save_images(f"cartoonset100k{'-rgb' if asrgb else ''}-{sdir.stem}-{SIZE}.gz", mnist, asrgb=asrgb, clast=True)
    labels = np.array(mnist_labels, dtype=np.uint8).reshape(-1)
    mio.save_labels(f"cartoonset100k-labels-{sdir.stem}.gz", labels)
    # df = pd.DataFrame(data=mnist_labels)
    # df.to_csv(f"cartoonset100k-{sdir.stem}.csv", index=False)


def load_cartoonset100k(asrgb=False):
    DATA_DIR = "E:\Datasets\CartoonSet\cartoonset100k"
    data_dir = path(DATA_DIR)
    # for sdir in data_dir.dirs():
    #     load_cartoonset100k_dir(sdir, asrgb=asrgb)
    Parallel(n_jobs=10)(delayed(load_cartoonset100k_dir)(sdir, asrgb) for sdir in data_dir.dirs())
    # end


def main():
    load_cartoonset10k(False)
    load_cartoonset100k(False)
    # load_cartoonset10k(True)
    # load_cartoonset100k(True)
    pass


if __name__ == "__main__":
    main()
