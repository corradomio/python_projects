from pathlib import Path
from random import uniform
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import cv2

import mitsubax as mix
from stdlib.tprint import tprint
from stdlib import csvx, kwargs

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
import mitsubax.sensors.cropped.wideangle_sensor
# import mitsubax.sensors.fullframe.wideangle_sensor
# wideangle-

TOTAL_CHICKENS = 2500
TOTAL_IMAGES = 1010
N_WEEKS = 6


def find_start_image_id():
    maxid = 1
    images = Path("refimgs")
    for ext in ["*.png", "*.jpg"]:
        for img in images.glob(ext):
            iid = int(img.stem[-4:])
            if iid >= maxid:
                maxid = iid+1
    return maxid


def gen_scene(sid: int, week: int, chicken_params):
    # mi.set_variant("cuda_ad_rgb")
    # import mitsubax.sensors.cropped.wideangle_sensor

    # -----------------------------------------------------------------------

    _, s, l, head_color, body_color, wing_color, count = chicken_params[week % N_WEEKS]
    w = 28      # m
    h = 12      # m

    scene_name = "wideangle_v2"

    nchicks = int(count * (w*h)/(80*12))
    tprint(f"nchicks: {nchicks}")

    # -----------------------------------------------------------------------

    tprint(f"[{sid}] load dict")

    kwargs = dict(
        s=s,
        head_color=head_color,
        body_color=body_color,
        wing_color=wing_color
    )
    scene_dict = mix.load_scene_dict(
        f"{scene_name}.xml",
        **kwargs
    )

    u2d = mix.Uniform2D(w,h,l)

    # -----------------------------------------------------------------------

    tprint(f"[{sid}] add chickens ...")
    cids = []

    # cid, zrot, xpos, ypos, sx, sy
    chicks_params = []
    for i in range(nchicks):
        cid = f"c@{i+1}"

        zrot = uniform(0, 360)
        xpos, ypos = u2d.sample()

        sx = uniform(0.9, 1.1)      #
        sy = uniform(0.9, 1.1)

        t = mix.ToWorld().scale(x=sx, y=sy, z=1).rotate(z=1, angle=zrot).translate(xpos, ypos).get()

        rh = uniform(0, 360)
        mix.clone(scene_dict, id=cid, ref="chicken", to_world=t, sgt={
            # "body": mix.ToWorld().rotate(x=1, angle=rh),
            # "head": mix.ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
        })

        # chicken = mix.get_by_id(scene_dict, cid)
        # pass

        cids.append(cid)
        chicks_params.append([cid, zrot, xpos, ypos, sx, sy])
    # end

    # -----------------------------------------------------------------------

    tprint(f"[{sid}] load scene")
    scene: mi.Scene = mix.load_dict(scene_dict)

    tprint(f"[{sid}] render ...")
    image: np.ndarray = mix.render(scene)

    # -----------------------------------------------------------------------

    # darker images
    # if sid%4 == 0:    # %25 % of darker images
    #     image = image*uniform(0.5, 1.0)

    tprint(f"[{sid}] save ...")
    ifile = f"images/{scene_name}-{sid:04}.png"
    plt.imsave(ifile, image)

    pfile = f"params/{scene_name}-{sid:04}.csv"
    csvx.dump(chicks_params, pfile, header=["cid", "zrot", "xpos", "ypos", "sx", "sy"])

    # -----------------------------------------------------------------------

    tprint(f"[{sid}] labels ...")
    sensor: mix.ForwardSensor = scene.sensors()[0]
    refimg = (image*255).astype(np.uint8)
    labels = []

    for cid in cids:
        cx,cy,w,h = mix.sensor_bbox(sensor, scene_dict, cid, format="cxcywh")
        if w == 0 or h == 0: continue

        labels.append([0, cx,cy,w,h])

        w2 = w/2
        h2 = h/2
        xmin = int(cx-w2)
        xmax = int(cx+w2)
        ymin = int(cy-h2)
        ymax = int(cy+h2)

        cv2.rectangle(refimg, (xmin,ymin), (xmax, ymax), (255,0,0), 2)
        pass

    lfile = f"labels/{scene_name}-{sid:04}.txt"
    csvx.dump(labels, lfile, header=None, delimiter=' ')

    ifile = f"refimgs/{scene_name}-{sid:04}.png"
    plt.imsave(ifile, refimg)

    # -----------------------------------------------------------------------

    del scene
# end


def main():
    ROOT = Path(".")
    for sdir in ["images", "params", "refimgs", "labels"]:
        (ROOT / sdir).mkdir(exist_ok=True, parents=True)

    chicken_params = csvx.load("chicken.csv", dtype=[int, float, float, str, str, str, int], skiprows=1)

    tprint("start ...")
    n = 10
    s = find_start_image_id()

    for i in range(s, s+n):
        tprint(f"...{i:4}/{n}")
        week = i
        gen_scene(i, week, chicken_params)
        # break
    pass

    # kwargs = dict( w=28, h=12, s=0.22)
    # Parallel(n_jobs=2)(delayed(gen_scene)(i, **kwargs) for i in range (s, n+1))
    pass


if __name__ == "__main__":
    main()
