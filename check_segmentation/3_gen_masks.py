from PIL import Image
import numpy as np
from path import Path as path
from stdlib.tprint import tprint
from joblib import Parallel, delayed
from commons import *


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


#
#   0 -> background
#   1 -> water drop
#   2 -> table
#   3 -> dispenser
#

def generate_mask(scene_file: path):
    tprint(f"... {scene_file.stem}")

    dir: path = scene_file.parent

    scene_id = idof(scene_file.stem)
    seg_file = dir / f"drop_segmentation-{scene_id}.png"
    mask_file = dir / f"drop_mask-{scene_id}.png"

    img0 = Image.open(seg_file)
    seg_image: np.ndarray = np.array(img0)

    # print(seg_image.shape)
    # plt.imshow(seg_image)
    # plt.show()

    img = seg_image
    dims: tuple[int] = img.shape[0:2]
    seg_mask = np.ones(dims, dtype=np.int8)

    # green: water drop
    # sloc = np.where((img[:,:,0] == 0) & (img[:,:,1] > 0) & (img[:,:,2] == 0))
    sloc = np.where((img[:, :, 0] < img[:, :, 1]) & (img[:, :, 1] > 0) & (img[:, :, 2] < img[:, :, 1]))
    seg_mask[sloc] = 2

    # blue: table
    # sloc = np.where((img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] > 0))
    sloc = np.where((img[:, :, 0] < img[:, :, 2]) & (img[:, :, 1] < img[:, :, 2]) & (img[:, :, 2] > 0))
    seg_mask[sloc] = 3

    # red: dispenser
    # sloc = np.where((img[:,:,0] > 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0))
    sloc = np.where((img[:, :, 0] > 0) & (img[:, :, 1] < img[:, :, 0]) & (img[:, :, 2] < img[:, :, 0]))
    seg_mask[sloc] = 4

    # done
    # plt.imshow(seg_mask*255/3)
    # plt.show()
    img = Image.fromarray(seg_mask, mode="L")
    img.save(mask_file)
    # iio.imwrite(mask_file, seg_mask, mode="L")


def generate_masks():
    images_dir = IMAGES_RESIZED
    for idir in images_dir.dirs():
        if idir.stem != "0000":
            continue
        tprint(idir, force=True)
        for scene_file in idir.files("drop_scene-*.png"):
            generate_mask(scene_file)
        # Parallel(n_jobs=10)(delayed(generate_mask)(scene_file)
        #                     for scene_file in idir.files("drop_scene-*.png"))
    # end
    tprint("done", force=True)
# end


def main():
    # print(ski.io.find_available_plugins())
    generate_masks()




if __name__ == "__main__":
    main()
