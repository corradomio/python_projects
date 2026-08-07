import matplotlib.pyplot as plt
import mitsuba as mi
from stdlib.imathx import isqrt

import mitsubax as mix

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
import mitsubax.plain.wideangle_sensor


def add_random_cubes(scene_dict: dict, n: int):

    k = isqrt(n)
    ds = 10/k

    for i in range(n):
        ix = i % k
        iy = i // k

        cid = f"c@{i+1}"

        # x = uniform(-7.4, 7.4)
        # y = uniform(-7.4, 7.4)
        x = -4.5 + ix*ds
        y = -4.5 + iy*ds

        # t = mix.ToWorld().scale(value=[0.5,0.5,0.5]).scale(value=[side,side,side]).translate(value=[x,y,0]).get()
        t = mix.ToWorld().translate(value=[x,y,0]).get()

        mix.clone(scene_dict, cid, ref="cube0", to_world=t)
        pass

    return scene_dict

def main():

    scene_name = "cube"

    print("load ...")
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml")
    add_random_cubes(scene_dict, 81)

    scene = mix.load_dict(scene_dict)

    print("render ...")
    image = mix.render(scene)

    print("save ...")
    plt.imsave(f"{scene_name}.png", image)
    pass


if __name__ == "__main__":
    main()
