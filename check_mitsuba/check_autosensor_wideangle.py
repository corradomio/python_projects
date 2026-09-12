import matplotlib.pyplot as plt
import mitsuba as mi
from stdlib.imathx import isqrt
from random import uniform

import mitsubax as mix
import numpy as np

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
# import mitsubax.plain.wideangle_sensor
import mitsubax.sensors.cropped.wideangle_sensor
# import mitsubax.sensors..fullframe.wideangle_sensor


INITIAL_CHICKS = 20000


def add_random_cubes(scene_dict: dict, n: int, w:float, h: float):
    w = w/2
    h = h/2
    z = 0

    k = isqrt(n)
    ds = 10/k

    centers_coords = []
    cids = []

    for i in range(n):
        ix = i % k
        iy = i // k

        cid = f"c@{i+1}"

        x = uniform(-w, w)
        y = uniform(-h, h)

        centers_coords.append([x,y,z])

        # t = mix.ToWorld().scale(value=[0.5,0.5,0.5]).scale(value=[side,side,side]).translate(value=[x,y,0]).get()
        t = mix.ToWorld().translate(value=[x,y,0]).get()

        mix.clone(scene_dict, cid, ref="cube0", to_world=t)
        cids.append(cid)
        pass

    for cid in cids:
        # shape = scene_dict[cid]
        bbox = mix.bounding_box(scene_dict, cid)

    return scene_dict, centers_coords


def cube_vertices(c, s):
    x,y,z = c
    t = s/2

    cv = [
        [x - t, y - t, z],
        [x + t, y - t, z],
        [x + t, y + t, z],
        [x - t, y + t, z],

        [x - t, y - t, z + s],
        [x + t, y - t, z + s],
        [x + t, y + t, z + s],
        [x - t, y + t, z + s],
    ]

    cv = np.array(cv, dtype=np.float32).T
    return mi.Point3f(cv)


def gen_scene(w: int, h: int, z: int, s: float):
    scene_name = "autosensor_wideangle"

    nchicks = int((INITIAL_CHICKS * w * h) / (80 * 12))

    print("load dict")
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml", w=w, h=h, z=z, s=s)

    print(f"add {nchicks} chicks")
    _, cc_list = add_random_cubes(scene_dict, nchicks, w=w, h=h)

    # mix.save_scene_dict(f"{scene_name}@{w}-{h}-{z}.json", scene_dict)

    print("load scene")
    scene: mi.Scene = mix.load_dict(scene_dict)

    # top_camera: mitsubax.cropped.WideAngleCamera = mitsubax.get_by_id(scene, "top-camera")
    # for cc in cc_list:
    #     cv_list = cube_vertices(cc, s)
    #     pixel, _, valid = top_camera.project_point(cv_list)
    #     for i in range(8):
    #         if not valid[i]: continue
    #
    #         x, y = int(pixel.x[i]), int(pixel.y[i])
    #     pass
    # # end

    print("render ...")
    image: np.ndarray = mix.render(scene)

    print(image.shape, image.max())     # [1080, 1920, 4], 1.0  RGB???

    print("labels ...")
    top_camera: mitsubax.sensors.cropped.WideAngleCamera = mitsubax.get_by_id(scene, "top-camera")
    for cc in cc_list:
        cv_list = cube_vertices(cc, s)
        pixel, valid = top_camera.project_point(cv_list)
        for i in range(8):
            if not valid[i]: continue

            x, y = int(pixel.x[i]), int(pixel.y[i])

            image[y,x,:]= [1,0,0,1]
        pass
    # end

    print("save ...")
    plt.imsave(f"{scene_name}@{w}-{h}-{z}.png", image)
# end


def main():
    global INITIAL_CHICKS
    INITIAL_CHICKS = 20000
    z = 4

    # gen_scene(28, 12, 4, 0.3)
    # gen_scene(20, 12, 4, 0.3)

    gen_scene(28, 12, 3, 0.1)
    gen_scene(20, 12, 3, 0.1)
    pass


if __name__ == "__main__":
    main()
