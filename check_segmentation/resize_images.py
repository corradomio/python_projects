from path import Path as path
from stdlib.tprint import tprint
from PIL import Image
from joblib import Parallel, delayed


IMAGES_ROOT = r"E:\Datasets\WaterDrop\orig"
IMAGES_RESIZED = r"E:\Datasets\WaterDrop\resized"

RESIZE_FACTOR = 2


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


def resize_image(ifile: path):
    tprint(ifile)
    iid = idof(ifile.stem)
    idir = int(iid)//1000
    im0 = Image.open(ifile)
    # print(im0.size)
    width, height = im0.size
    im1 = im0.resize((width//RESIZE_FACTOR, height//RESIZE_FACTOR))
    # print(im1.size)

    rdir = path(f"{IMAGES_RESIZED}\\{idir:04}")
    rdir.makedirs_p()
    rfile = rdir / ifile.name

    im1.save(rfile)


def copy_json(jfile):
    tprint(jfile)
    iid = idof(jfile.stem)
    idir = int(iid) // 1000

    cdir = path(f"{IMAGES_RESIZED}\\{idir:04}")
    cfile = cdir / jfile.name

    jfile.copy(cfile)


def main():
    iroot = path(IMAGES_ROOT)
    for idir in iroot.dirs():
        tprint(idir, force=True)
        for ifile in idir.files("*.png"):
            tprint(ifile)
            if "drop_scene" in ifile:
                resize_image(ifile)
            elif "drop_segmentation" in ifile:
                resize_image(ifile)
        for jfile in idir.files("*.json"):
            copy_json(jfile)
    # end
# end


if __name__ == "__main__":
    main()
