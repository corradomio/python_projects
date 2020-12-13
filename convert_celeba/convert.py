import cv2
from joblib import Parallel, delayed
from path import Path as path

CELEBA_HOME_DIR = "E:/Datasets/CelebA/"
IMG_SHAPE = (218, 178, 3)


def color(dim):
    print("color", dim)
    img_align_celeba = path(CELEBA_HOME_DIR + "img_align_celeba")

    img_dir = path(CELEBA_HOME_DIR + "color_{}".format(dim))
    img_dir.mkdir_p()

    for img_file in img_align_celeba.files("*.jpg"):
        img_name = img_file.basename()
        img = cv2.imread(img_file)
        img = cv2.resize(img, (dim, dim))

        cvt_name = img_dir.joinpath (img_name)
        cv2.imwrite(cvt_name, img)


def gray(dim):
    print("gray", dim)
    img_align_celeba = path(CELEBA_HOME_DIR + "img_align_celeba")

    img_dir = path(CELEBA_HOME_DIR + "gray_{}".format(dim))
    img_dir.mkdir_p()

    for img_file in img_align_celeba.files("*.jpg"):
        img_name = img_file.basename()
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (dim, dim))

        cvt_name = img_dir.joinpath (img_name)
        cv2.imwrite(cvt_name, img)


def call(f, size): f(size)


def main():
    functions = [color, gray]
    dims = [64, 32]

    Parallel(n_jobs=10)(delayed(call)(f, dim) for f in functions for dim in dims)
# end


if __name__ == "__main__":
    main()
