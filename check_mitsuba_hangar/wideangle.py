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

def find_start_image_id():
    maxid = 1
    images = Path("images")
    for ext in ["*.png", "*.jpg"]:
        for img in images.glob(ext):
            iid = int(img.stem[10:])
            if iid >= maxid:
                maxid = iid+1
    return maxid


def gen_scene(sid: int, **kwargs):
    mi.set_variant("cuda_ad_rgb")
    import mitsubax.sensors.cropped.wideangle_sensor

    scene_name = "wideangle"

    nchicks = TOTAL_CHICKENS

    tprint("load dict")
    scene_dict = mix.load_scene_dict(
        f"{scene_name}.xml",
        **kwargs
    )

    w = kwargs["w"]
    h = kwargs["h"]
    s = kwargs["s"]
    u2d = mix.Uniform2D(w,h,s)

    tprint("add chickens ...")
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
        if i%2== 0:
            mix.clone(scene_dict, id=cid, ref="adult-chicken", to_world=t, sgt={
                # "body": ToWorld().scale(x=bsx, y=bsy, z=bsx).get(),
                # "head": ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
            })
        else:
            mix.clone(scene_dict, id=cid, ref="young-chicken", to_world=t, sgt={
                # "body": ToWorld().scale(x=bsx, y=bsy, z=bsx).get(),
                # "head": ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
            })


        cids.append(cid)
        chicks_params.append([cid, zrot, xpos, ypos, sx, sy])
    # end

    tprint("load scene")
    scene: mi.Scene = mix.load_dict(scene_dict)

    # -----------------------------------------------------------------------

    tprint("render ...")
    image: np.ndarray = mix.render(scene)
    tprint(f"... {image.shape} [{image.min()}, {image.max()}]")

    # image = image*0.5
    # goldenhour = np.array(
    #     [
    #         [ 1.25, 0.10, 0.00],
    #         [ 0.05, 1.05, 0.00],
    #         [-0.10,-0.05, 0.75]
    #     ]
    # )
    #
    # alba = np.array(
    #     [
    #         [ 1.15, 0.05,-0.05],
    #         [ 0.10, 0.95, 0.00],
    #         [-0.10, 0.05, 0.85]
    #     ]
    # )
    #
    # tramonto = np.array(
    #     [
    #         [ 1.35, 0.10,-0.10],
    #         [ 0.10, 0.80,-0.05],
    #         [-0.15,-0.05, 0.60]
    #     ]
    # )

    # image = golden_hour @ image
    # image = np.tensordot(image, goldenhour, axes=([2], [0]))
    # image = (image.reshape(-1, 3) @ tramonto).reshape(image.shape).clip(0, 1)

    tprint("save ...")
    ifile = f"images/{scene_name}-{sid:04}.png"
    plt.imsave(ifile, image)

    pfile = f"params/{scene_name}-{sid:04}.csv"
    csvx.dump(chicks_params, pfile, header=["cid", "zrot", "xpos", "ypos", "sx", "sy"])

    # -----------------------------------------------------------------------
    tprint("labels ...")
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

    ifile = f"refimgs/{scene_name}-{sid:04}.png"
    plt.imsave(ifile, refimg)

    lfile = f"labels/{scene_name}-{sid:04}.csv"
    csvx.dump(labels, lfile, header=None)

    # -----------------------------------------------------------------------

    del scene
# end


def main():
    tprint("start ...")
    n = 1000
    s = find_start_image_id()

    # for i in range(s, n):
    #     tprint(f"...{i:4}/{n}")
    #     gen_scene(i, w=28, h=12, s=0.22)
    # pass

    kwargs = dict( w=28, h=12, s=0.22)
    Parallel(n_jobs=3)(delayed(gen_scene)(i, **kwargs) for i in range (s,s+n))


if __name__ == "__main__":
    main()
