#
# Some IO utilities
# Version 2.0
#
# http://yann.lecun.com/exdb/mnist/
#
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
#
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
#
#
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
#
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
#


import struct
import numpy as np
import matplotlib.pyplot as plt
import gzip
from math import sqrt
from path import Path as path


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def load_images(images_file: str, count=0, asfloat=False, asvector=False, asrgb=False) -> np.ndarray:
    """
    Load the images in a MNIST formatted file.

    A RGB image is composed of 3 consecutive images: red, green, blue.
    The parameter 'asrgb' specify if to compose 3 B/W images in a color image

    :param str images_file: file to load
    :param int count: number of images to load. 0 for all
    :param bool asfloat: if to convert (True) the pixel colors in the range [0,1]
    :param bool asvector: if to consider a image as a 1D (true) vector or a 2D matrix
    :param bool asrgb: if the image is in RGB

    :return np.ndarray: the loaded images as array (count,w,h) or (count, w*h)
    """
    def torgb(imgs):
        if len(imgs.shape) == 2:
            nimages, isize = imgs.shape
            imgs = imgs.reshape((nimages // 3, 3, isize))
            imgs = np.swapaxes(imgs, 1, 2)
        else:
            nimages, height, width = imgs.shape
            imgs = imgs.reshape((nimages // 3, 3, height, width))
            imgs = np.swapaxes(imgs, 1, 2)
            imgs = np.swapaxes(imgs, 2, 3)
        return imgs

    def read32(f):
        raw_int = f.read(4)
        return struct.unpack(">l", raw_int)[0]

    if images_file.endswith(".gz"):
        f = gzip.open(images_file, mode="rb")
    else:
        f = open(images_file, mode="rb")

    # with open(mnist_file, mode="rb") as f:
    with f:
        magic = read32(f)
        # check if the file is readed correctly
        assert magic == 2051

        n = read32(f)
        w = read32(f)
        h = read32(f)

        if asrgb:
            count *= 3

        if 0 < count < n:
            n = count

        data = f.read(n*w*h)
        images = np.frombuffer(data, dtype=np.uint8)
        """:type: np.ndarray"""

    # image structure: 1D or 2D
    if asvector:
        images = images.reshape((n, h*w))
    else:
        images = images.reshape((n, h, w))

    #
    if asrgb:
        images = torgb(images)
    if asfloat:
        images = images.astype('float32') / 255.
    return images
# end


def save_images(images_file: str, images: np.ndarray, asrgb=False, clast=False):
    """
    Save the images in a file in MNIST format

    :param str mnist_file: file to save the images
    :param bool asrgb: if to save the image as RGB
    :param bool clast: if the channels are the last dimension
    :param np.ndarray images: images to save
    """

    def isqrt(i):
        return int(sqrt(float(i)))

    def write32(f, i):
        raw_int = struct.pack(">l", i)
        f.write(raw_int)

    # if the images are in the range [0,1], convert them
    if images.dtype != np.uint8:
        images = (images.clip(0, 1) * 255).astype(np.uint8)

    if asrgb:
        if clast:
            images = images.transpose([0, 3, 1, 2])

        if len(images.shape) == 2:
            n, s = images.shape
            n *= 3
            s //= 3
            h = w = isqrt(s)
        else:
            n, c, h, w = images.shape
            n *= c
    else:
        if len(images.shape) == 2:
            n, s = images.shape
            h = w = isqrt(s)
        else:
            n, h, w = images.shape

    if images_file.endswith(".gz"):
        f = gzip.open(images_file, mode="wb")
    else:
        f = open(images_file, mode="wb")

    # with open(images_file, mode="wb") as f:
    with f:
        write32(f, 2051)    # magick
        write32(f, n)       # number of images
        write32(f, h)       # number of rows (height)
        write32(f, w)       # number of cols (width)
        data = images.tobytes()
        f.write(data)
# end


def load_labels(labels_file: str, asrgb=False):
    """
    Load the labels of the MNIST images

    :param str labels_file:
    :return np.ndarray: array of labels
    """

    def read32(f):
        raw_int = f.read(4)
        return struct.unpack(">l", raw_int)[0]

    if labels_file.endswith(".gz"):
        f = gzip.open(labels_file, mode="rb")
    elif labels_file.endswith(".zip"):
        f = zip.open
    else:
        f = open(labels_file, mode="rb")

    # with open(label_file, mode="rb") as f:
    with f:
        magic = read32(f)
        # check if the file is correctly
        assert magic == 2049

        n = read32(f)

        raw_labels = f.read(n)
        labels = np.frombuffer(raw_labels, dtype=np.uint8)
        """:type: np.ndarray"""

    if asrgb:
        labels = [labels[i] for i in range(0, len(labels), 3)]

    return labels.astype(dtype='uint8')
# end


def save_labels(labels_file: str, labels: np.ndarray, asrgb=False):
    """
    Save the labels in a file in MNIST format
    :param str labels_file:
    :param np.ndarray labels: labels
    """

    def write32(f, i):
        raw_int = struct.pack(">l", i)
        f.write(raw_int)

    n = len(labels)

    if asrgb:
        labels3 = []
        for l in labels:
            labels3 += [l, l, l]
        labels = labels3
        n *= 3

    if labels_file.endswith(".gz"):
        f = gzip.open(labels_file, mode="wb")
    else:
        f = open(labels_file, mode="wb")

    # with open(labels_file, mode="wb") as f:
    with f:
        write32(f, 2049)    # magick
        write32(f, n)       # number of images

        data = np.array(labels).tobytes()
        f.write(data)
# end


# ---------------------------------------------------------------------------

def load_images_csv(csv_file, skiprows=0, delimiter=','):
    images = []
    with open(csv_file) as f:
        for s in range(skiprows):
            next(f)
        for r in f:
            img = list(map(lambda i: int(i), r.split(delimiter)))
            images.append(img)
    images = np.array(images, dtype=float)
    imax = images.max()
    if imax < 1.1:
        images = (images*255).astype('uint8')
    elif imax > 255:
        images = (images*(255/imax)).astype('uint8')
    else:
        images = images.astype('uint8')

    n, m = images.shape
    if m != 784:
        raise TypeError("Invalid MNIST image size: {} != 784".format(m))

    return images.reshape((n, 28, 28))
# end


def save_images_csv(csv_file, images, labels=None):
    """Save the images in a CSV file

    :param str csv_file: file where to save the images
    :param np.ndarray images: images
    :param list|np.ndarray labels: label assigned to each image
    :return:
    """

    shape = images.shape
    if len(shape) == 3:
        n, w, h = shape
        s = h*w
        images = images.reshape(n, s)
    else:
        n, s = shape

    def istr(x): return str(int(x))

    def fstr(x): return str(float(x))

    tostr = fstr if images.dtype in [float, np.float, np.float16, np.float32, np.float64] else istr

    hdr = ["c{0:03}".format(i+1) for i in range(s)]

    with open(csv_file, mode="w") as f:

        # write the header
        if labels is not None:
            hdr = hdr + ["label"]
        f.write(",".join(hdr) + "\n")

        for i in range(n):
            data = list(images[i, :])
            line = ",".join(map(tostr, data))

            if labels is not None:
                line = line + ",{0}".format(labels[i])
            f.write(line + "\n")
        pass
    pass
# end


def load_labels_csv(csv_file, skiprows=0):
    labels = []
    with open(csv_file) as f:
        for s in range(skiprows):
            next(f)
        for r in f:
            l = int(r)
            labels.append(l)
    return np.array(labels)
# end


def save_labels_csv(csv_file, labels):
    """Save the labels in a CSV file

    :param str csv_file: file where to save the images
    :param list labels: labels to save
    :return:
    """
    n = len(labels)

    with open(csv_file, mode="w") as f:
        f.write("label\n")

        for i in range(n):
            f.write(str(i) + "\n")
        pass
    pass
# end


# ---------------------------------------------------------------------------

def show_images(images, cols=1, title=None):
    """
    Display a list of images in a single figure with matplotlib

    Note: it is possible to write:

        show_images(...).show()
        show_images(...).savefig(...)

    :param np.ndarray images: images to plot
    :param int cols: number of columns to use
    :param str title: title of the plot
    """

    def isqrt(i):
        return int(sqrt(float(i)))

    if images.dtype != np.uint8:
        images = (images*255.).astype(np.uint8)

    if len(images.shape) == 2:
        n, s = images.shape
        w = isqrt(s)
        images = images.reshape((n, w, w))

    n = len(images)
    rows = max((n+cols-1)//cols, 1)

    titles = [""]*n
    fig = plt.figure(figsize=(rows, cols))
    if title is not None: fig.suptitle(title)

    for k, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, cols, k + 1)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.axis('off')
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    # class ShowThis:
    #     def show(self):
    #         plt.show()
    #
    #     def savefig(self, *args, **kwargs):
    #         plt.savefig(*args, **kwargs)
    #         plt.close()
    #
    # return ShowThis()
# end


def show_comparisons(list_images, y_labels= None, title=None):
    """
    Display a list of images. Each row contains a different set of images

    Note that images is a LIST of arrays!

    :param list[np.ndarray] list_images: the images to show
    :param str title: title of the plot
    """
    assert type(list_images) == list

    def isqrt(i):
        return int(sqrt(float(i)))

    nrows = len(list_images)
    ncols = len(list_images[0])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)

    if title is not None:
        fig.suptitle(title)

    plt.yticks([])
    for r in range(nrows):
        images = list_images[r]

        if images.dtype != np.uint8:
            images = (images * 255.).astype(np.uint8)

        if len(images.shape) == 2:
            n, s = images.shape
            w = isqrt(s)
            images.reashape((n, w, w))

        for c in range(ncols):
            image = images[c]

            # a = fig.add_subplot(nrows, ncols, r*ncols+c+1)
            a = axes[r, c]
            a.set_aspect(1, adjustable='box-forced')
            a.axes.get_xaxis().set_visible(False)
            a.axes.get_yaxis().set_visible(False)
            a.imshow(image, cmap="gray", vmin=0, vmax=255)
            if c == 0:
                a.axes.get_yaxis().set_visible(True)
                if y_labels and y_labels[r]:
                    a.set_ylabel(y_labels[r])
    # fig.tight_layout()
    plt.axis('off')
    # plt.tight_layout()
    return fig
# end


def view_images(mnist_file, rows=5, cols=5, asrgb=False, title=None):
    """As show_images, but load also the images

    :param str mnist_file: file to load
    :param rows: n of rows to visualize
    :param cols: n of columns to visualize
    :param asrgb: if the images as in RGB format
    :param title: title of the ficure
    :return:
    """
    n = rows*cols
    images = load_images(mnist_file, count=n, asrgb=asrgb)
    if title is None:
        title = mnist_file
    show_images(images, cols=cols, title=title)
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def binarize_images(images, mode=0.5):
    """
    Binarize the image in {0,1}

    :param np.array images: array of images
    :param mode: how to binarize. Values: 'mean', 'mean/2' (half mean) or a float value
    :return np.array[np.uint8]: the binarized images
    """
    if images.dtype == np.float32:
        pass
    if images.dtype == np.uint8:
        images = images.astype('float32') / 255.
    if images.max() > 2:
        images = images / images.max()

    value = 0.5
    if mode == "mean":
        value = images.mean()
    if mode == "mean/2":
        value = images.mean()/2.0
    if type(mode) in [float, np.float32]:
        value = mode

    # the images are in the range [0,1]
    images[images <= value] = 0.0
    images[images >= value] = 1.0
    return (images*255).astype('uint8')
# end


def scale_images(images, pixel_size, mode="mean", rescale=True):
    """Rescale the images converted multiple pixels in a single pixels
    It is possible to select how to convert the pixel color:

        - 'max':  it is used the maximum value for the color
        - 'mean': it is computed the mean color between all pizels

    :param np.ndarray images: NxHxW images
    :param int pixel_size: n of pixels
    :param str mode: how to compose the pixels:
                        - "mean"
                        - "max"
    :param bool rescale: to ensure that the brightest pixel has color 255
    :return np.ndarray: the scaled images
    """
    def mean(matrix):
        h, w = matrix.shape
        return matrix.sum()/(h*w)

    def max(matrix):
        return matrix.max()

    scaler = mean if mode in ["mean", False] else max

    ps = pixel_size
    n, ho, wo = images.shape        # original image sizes
    n, hs, ws = n, ho//ps, wo//ps   # scaled image sizes

    scaled = np.zeros(shape=(n, hs, ws), dtype=images.dtype)

    for i in range(n):
        for r in range(hs):
            for c in range(ws):
                y = r*ps
                x = c*ps
                pixel = scaler(images[i, y:y+ps, x:x+ps])
                scaled[i, r, c] = pixel
            pass
        pass
        if rescale:
            scaled[i] = (scaled[i].astype(int)*255/scaled[i].max()).astype(images.dtype)
    pass
    return scaled
# end


def solarize_images(images, n_levels=4):
    """
    Reduce the number of colors in the image

    :param np.ndarray images: list of images
    :param int n_levels: n of color
    :return: solarized images
    """
    assert n_levels > 1

    if images.dtype in [np.float64]:
        # values in the range [0, 1]
        assert images.max() <= 1.0

        delta = 1.0/(n_levels-1)
        error = delta/2.
        pass
    else:
        # values in the range [0, 255]
        assert images.max() <= 255

        delta = 255//(n_levels-1)
        error = delta//2
    pass

    solarized = (((images + error)//delta)*delta).astype(images.dtype)
    return solarized
# end


# ---------------------------------------------------------------------------

def rowcol_images(images, mode=None):
    """
    Aggregate the rows and the columns of the 'binarized' image in a integer with w or h bits

    :param np.ndarray images: array of binarized images
    :param mode: how to convert the image:
                    None:           [col1,... row1, ...]
                    "col",True:     [col1,... ]
                    "row",False:    [row1,... ]
    :return: the converted images
    """

    def isqrt(i):
        return int(sqrt(float(i)))

    def powers(n):
        assert n <= 32
        p = np.zeros(shape=n)
        e = 1
        for i in range(n):
            p[i] = e
            e = e*2
        return p

    def toint(ary, pow):
        s = int(np.sum(ary*pow))
        return s
    pass

    if images.dtype == np.float32:
        pass
    if images.dtype == np.uint8:
        images = images.astype('float32') / 255.
    if images.max() > 2:
        images = images / images.max()

    if len(images.shape) == 2:
        n, s = images.shape
        w = isqrt(s)
        h = s//w
        images = images.reshpe((n, h, w))
    else:
        n, h, w = images.shape
    pass

    if mode in ["row", False]:
        mode = False
        s = h
    elif mode in ["col", True]:
        mode = True
        s = w
    else:
        s = w + h

    rowcols = np.zeros(shape=(n, s), dtype='uint32')

    for i in range(n):
        j = 0

        # col
        if mode is None or mode is True:
            coeff = powers(h)
            for c in range(h):
                row = images[i, :, c]
                val = toint(row, coeff)
                rowcols[i, j] = val
                j += 1
        pass

        # row
        if mode is None or mode is False:
            coeff = powers(w)
            for r in range(w):
                col = images[i, r, :]
                val = toint(col, coeff)
                rowcols[i, j] = val
                j += 1
        pass

    pass

    return rowcols
# end


def generate_thumbnails(dir, shape=(5, 5)):

    print("processing", dir)

    h, w = shape
    n = h*w

    def _generate_thumbnail(mnist_file):
        print("   ", mnist_file.namebase)
        images = load_images(mnist_file, count=n)
        show_images(images, w)
        plt.savefig(str(mnist_file) + ".png", dpi=300)
    # end

    dir = path(dir)
    for fimage in dir.files("*-idx3-ubyte"):
        _generate_thumbnail(fimage)
# end
