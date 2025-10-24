from path import Path as path
from stdlib.tprint import tprint
from PIL import Image
from joblib import Parallel, delayed
from commons import *


CROP_SIZE = 224
RESIZE_FACTOR = 4


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


def resize_image(ifile: path):
    tprint(ifile)
    iid = idof(ifile.stem)
    idir = int(iid)//1000
    im0 = Image.open(ifile)
    width, height = im0.size

    woffset = (width - CROP_SIZE)//2.
    hoffset = (height - CROP_SIZE)//2.

    im1 = im0.crop((woffset, hoffset, woffset+CROP_SIZE, hoffset+CROP_SIZE))
    width, height = im1.size
    im2 = im1.resize((width//RESIZE_FACTOR, height//RESIZE_FACTOR))

    rdir = path(f"{IMAGES_RESIZED}/{idir:04}")
    rdir.makedirs_p()
    rfile = rdir / ifile.name

    im2.save(rfile)


def copy_json(jfile):
    tprint(jfile)
    iid = idof(jfile.stem)
    idir = int(iid) // 1000

    cdir = path(f"{IMAGES_RESIZED}/{idir:04}")
    cfile = cdir / jfile.name

    jfile.copy(cfile)


def crop_image(ifile):
    # tprint(ifile)
    if ".json" in ifile:
        copy_json(ifile)
    elif "drop_scene" in ifile:
        resize_image(ifile)
    elif "drop_segmentation" in ifile:
        resize_image(ifile)
    else:
        pass
# end


def main():
    iroot = path(IMAGES_ROOT)
    for idir in iroot.dirs():
        if idir.stem != "0000":
            continue
        (IMAGES_RESIZED / idir.stem).makedirs_p()
        tprint(idir, force=True)
        # for ifile in idir.files():
        #     crop_image(ifile)

        Parallel(n_jobs=12)(delayed(crop_image)(ifile) for ifile in idir.files())
    # end
# end


if __name__ == "__main__":
    main()
