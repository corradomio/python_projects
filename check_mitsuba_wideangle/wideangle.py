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


def find_start_image_id():
    maxid = 1
    images = Path("refimgs")
    for ext in ["*.png", "*.jpg"]:
        for img in images.glob(ext):
            iid = int(img.stem[-4:])
            if iid >= maxid:
                maxid = iid+1
    return maxid


def load_chicken_params():
    # week, length,side, head,body,wing, count
    chicken_params = csvx.load("chicken.csv", dtype=[int,float,float, str,str,str,int], skiprows=1)
    pass



def gen_scene(sid: int, **kwargs):
    # mi.set_variant("cuda_ad_rgb")
    # import mitsubax.sensors.cropped.wideangle_sensor

    scene_name = "wideangle_v2"

    nchicks = TOTAL_CHICKENS

    tprint(f"[{sid}] load dict")
    scene_dict = mix.load_scene_dict(
        f"{scene_name}.xml",
        **kwargs
    )

    w = kwargs["w"]
    h = kwargs["h"]
    s = kwargs["s"]
    u2d = mix.Uniform2D(w,h,s)

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

        # mix.instance(scene_dict, id=cid, ref="chicken", to_world=t)
        if i%4 == 0: # 25% of youg chickens
            mix.clone(scene_dict, id=cid, ref="young-chicken", to_world=t, sgt={
                # "body": ToWorld().scale(x=bsx, y=bsy, z=bsx).get(),
                # "head": ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
            })
        else:
            mix.clone(scene_dict, id=cid, ref="adult-chicken", to_world=t, sgt={
                # "body": ToWorld().scale(x=bsx, y=bsy, z=bsx).get(),
                # "head": ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
            })

        cids.append(cid)
        chicks_params.append([cid, zrot, xpos, ypos, sx, sy])
    # end

    tprint(f"[{sid}] load scene")
    scene: mi.Scene = mix.load_dict(scene_dict)

    # -----------------------------------------------------------------------

    tprint(f"[{sid}] render ...")
    image: np.ndarray = mix.render(scene)

    # darker images
    if sid%4 == 0:    # %25 % of darker images
        image = image*uniform(0.5, 1.0)

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

    chicken_params = load_chicken_params()

    tprint("start ...")
    n = TOTAL_IMAGES
    s = find_start_image_id()

    for i in range(s, n):
        tprint(f"...{i:4}/{n}")
        gen_scene(i, s=0.22)
        break
    pass

    # kwargs = dict( w=28, h=12, s=0.22)
    # Parallel(n_jobs=2)(delayed(gen_scene)(i, **kwargs) for i in range (s, n+1))
    pass


if __name__ == "__main__":
    main()
