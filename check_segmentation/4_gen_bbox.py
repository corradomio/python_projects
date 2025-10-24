from PIL import Image
import numpy as np
from math import pi
from path import Path as path
from stdlib.tprint import tprint
from joblib import Parallel, delayed
from stdlib.jsonx import load, dump
from commons import IMAGES_ROOT


def sq(x): return x*x

def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


#
#   0 -> background
#   1 -> water drop
#   2 -> table
#   3 -> dispenser
#
# Note: in the image the values used are [1-4]
#


def bbox_image(scene_file: path):
    try:
        tprint(f"... {scene_file.stem}")

        dir: path = scene_file.parent

        scene_id = idof(scene_file.stem)
        mask_file = dir / f"drop_mask-{scene_id}.png"

        im0 = Image.open(mask_file)
        width, height = im0.size
        mask_image: np.ndarray = np.array(im0)-1

        # print(mask_image.min(), mask_image.max(), mask_image.shape, mask_image.dtype)
        #
        #  WARNING: FIRST COLUMNTS, AFTER ROWS
        #
        y_locs, x_locs = np.where(mask_image == 1)
        if len(x_locs) > 0 and len(y_locs) > 0:
            x_locs.sort()
            y_locs.sort()
            xl, xr = int(x_locs[0]), int(x_locs[-1])
            yt, yb = int(y_locs[0]), int(y_locs[-1])
        # elif len(x_locs) > 0:
        #     tprint(f"... ... {scene_file.stem}: no height", force=True)
        #     x_locs.sort()
        #     xl, xr = int(x_locs[0]), int(x_locs[-1])
        #     yt, yb = 0, 1
        # elif len(y_locs) > 0:
        #     tprint(f"... ... {scene_file.stem}: no width", force=True)
        #     y_locs.sort()
        #     xl, xr = 0, 1
        #     yt, yb = int(y_locs[0]), int(y_locs[-1])
        else:
            tprint(f"... ... {scene_file.stem}: no drop", force=True)
            xl, xr = 0, 1
            yt, yb = 0, 1

        if xl == xr: xr = xl+1
        if yt == yb: yb = yt+1

        jfile = dir / f"drop_scene-{scene_id}.json"
        jdata = load(jfile)
        jdata["note"] = {
            "drop_bbox": "xmin, ymin, xmax, ymax",
            "image_size": "width, height"
        },
        jdata["drop_bbox" ] = [xl, yt, xr, yb]
        jdata["image_size"] = [width,  height]

        # volume
        drop_height = jdata["drop_height"]
        drop_radius = jdata["drop_radius"]
        drop_volume = pi * sq(drop_height) * (drop_radius - drop_height / 3)
        jdata["drop_volume"] = drop_volume

        dump(jdata, jfile)
    except Exception as ex:
        print(mask_file, ex)
        pass
# end


def main():
    # iroot = IMAGES_RESIZED
    iroot = IMAGES_ROOT
    for idir in iroot.dirs():
        # if idir.stem != "0000":
        #     continue
        tprint(idir, force=True)
        # for ifile in idir.files("drop_mask-*.png"):
        #     bbox_image(ifile)
        Parallel(n_jobs=12)(delayed(bbox_image)(ifile) for ifile in idir.files("drop_mask-*.png"))
    # end
    tprint("done", force=True)
# end



if __name__ == "__main__":
    main()
