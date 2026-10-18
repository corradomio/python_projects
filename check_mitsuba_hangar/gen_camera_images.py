from pathlib import Path
from stdlib import csvx

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from joblib import Parallel, delayed
import cv2

import mitsubax as mix
from gen_hangar import HANGAR_SIZE, CAMERAS_LAYOUT, CAMERA_SIZE, CHICKEN_TYPES
from gen_hangar import gen_hangar, Hangar, Camera
from stdlib.tprint import tprint

print(mi.variants())
mi.set_variant("cuda_ad_rgb")


def gen_camera_scene(sid, camera: Camera):
    mi.set_variant("cuda_ad_rgb")
    import mitsubax.sensors.perspective_sensor as ps

    scene_name = "hangar"
    nx, ny = CAMERAS_LAYOUT
    n = nx*ny

    cx, cy = camera.center
    kwargs = {"cx": cx, "cy": cy}

    tprint("load dict")
    scene_dict = mix.load_scene_dict(
        f"{scene_name}.xml",
        **kwargs
    )

    tprint("... add chickens")
    cids = []
    chicks_params = []
    for chicken in camera.chickens:
        cid = chicken.id

        ref = chicken.type
        zrot = chicken.rot
        xpos, ypos = chicken.center
        sx, sy = chicken.body_factor

        t = mix.ToWorld().scale(x=sx, y=sy, z=1).rotate(z=1, angle=zrot).translate(xpos, ypos).get()

        mix.clone(scene_dict, id=cid, ref=ref, to_world=t, sgt={
            # "body": ToWorld().scale(x=bsx, y=bsy, z=bsx).get(),
            # "head": ToWorld().scale(x=hsx, y=hsy, z=hsx).get(),
        })

        cids.append(cid)
        chicks_params.append([cid, ref, zrot, xpos, ypos, sx, sy])
    # end

    tprint("load scene")
    scene: mi.Scene = mix.load_dict(scene_dict)

    # -----------------------------------------------------------------------

    tprint(f"render {sid}/{n} ...")
    image: np.ndarray = mix.render(scene)

    tprint("save ...")
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

    lfile = f"labels/{scene_name}-{sid:04}.csv"
    csvx.dump(labels, lfile, header=None)

    ifile = f"refimgs/{scene_name}-{sid:04}.png"
    plt.imsave(ifile, refimg)

    # -----------------------------------------------------------------------

    del scene
# end




def main():
    ROOT = Path(".")
    for sdir in ["images", "params", "refimgs", "labels"]:
        (ROOT / sdir).mkdir(exist_ok=True, parents=True)

    hangar: Hangar = gen_hangar(HANGAR_SIZE, CAMERAS_LAYOUT, CAMERA_SIZE, CHICKEN_TYPES)
    hangar.statistics()
    hangar.plot_cameras()

    # for i, camera in enumerate(hangar.cameras.values()):
    #     gen_camera_scene(i+1, camera)

    Parallel(n_jobs=2)(delayed(gen_camera_scene)(i+1, camera) for i, camera in enumerate(hangar.cameras.values()))

    pass


if __name__ == "__main__":
    main()
