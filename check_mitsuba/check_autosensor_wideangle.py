import matplotlib.pyplot as plt
import mitsuba as mi
from stdlib.imathx import isqrt
from random import uniform

import mitsubax as mix

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
import mitsubax.plain.wideangle_sensor


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



def gen_scene(w: int, h: int, z: int, s: float):
    scene_name = "autosensor_wideangle"

    nchicks = int((20000 * w * h) / (80 * 12))
    print("n chicks:", nchicks)

    scene_dict = mix.load_scene_dict(f"{scene_name}.xml", w=w, h=h, z=z, s=s)
    add_random_cubes(scene_dict, nchicks, w=w, h=h)

    scene = mix.load_dict(scene_dict)

    print("render ...")
    image = mix.render(scene)

    print("save ...")
    plt.imsave(f"{scene_name}@{w}-{h}-{z}.png", image)
# end


def main():
    z = 4

    # gen_scene(28, 12, 4, 0.3)
    # gen_scene(20, 12, 4, 0.3)

    gen_scene(28, 12, 3, 0.3)
    gen_scene(20, 12, 3, 0.3)
    pass


if __name__ == "__main__":
    main()
