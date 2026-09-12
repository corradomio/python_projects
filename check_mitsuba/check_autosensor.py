import matplotlib.pyplot as plt
import mitsuba as mi
from stdlib.imathx import isqrt
from random import uniform

import mitsubax as mix

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
import mitsubax.sensors.circular.wideangle_sensor


def add_random_cubes(scene_dict: dict, n: int, w:float, h: float):
    w = w/2
    h = h/2

    k = isqrt(n)
    ds = 10/k

    for i in range(n):
        ix = i % k
        iy = i // k

        cid = f"c@{i+1}"

        x = uniform(-w, w)
        y = uniform(-h, h)

        # t = mix.ToWorld().scale(value=[0.5,0.5,0.5]).scale(value=[side,side,side]).translate(value=[x,y,0]).get()
        t = mix.ToWorld().translate(value=[x,y,0]).get()

        mix.clone(scene_dict, cid, ref="cube0", to_world=t)
        pass

    return scene_dict
# end


def gen_scene(w: int, h: int, z: int, s: float):
    scene_name = "autosensor"

    nchicks = int((20000 * w * h) / (80 * 12))
    print("n chicks:", nchicks)

    print("load ...")
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml", w=w, h=h, s=s, z=z)
    add_random_cubes(scene_dict, nchicks, w=w, h=h)

    scene = mix.load_dict(scene_dict)

    print("render ...")
    image = mix.render(scene)

    print("save ...")
    plt.imsave(f"{scene_name}@{w}-{h}-{z}.png", image)
# end


def main():

    # gen_scene(28, 2, 4, 0.3)
    # gen_scene(28, 3, 4, 0.3)
    # gen_scene(28,12, 4, 0.3)
    # gen_scene(20,12, 4, 0.3)
    # gen_scene(10, 6, 4, 0.3)

    gen_scene(28, 2, 3, 0.1)
    gen_scene(28, 3, 3, 0.1)
    gen_scene(28,12, 3, 0.1)
    gen_scene(20,12, 3, 0.1)
    gen_scene(10, 6, 3, 0.1)

    pass


if __name__ == "__main__":
    main()
