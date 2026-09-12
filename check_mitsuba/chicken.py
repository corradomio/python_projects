import matplotlib.pyplot as plt
import mitsuba as mi

import mitsubax as mix
from random import uniform

from mitsubax.mitsubax import bounding_box

print(mi.variants())
mi.set_variant("cuda_ad_rgb")


def gen_scene():
    scene_name = "chicken"

    print("load ...")
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml")

    cids = []

    for i in range(10):
        cid = f"c@{i+1}"
        zrot = uniform(-180,180)
        xpos = uniform(-2.5, 2.5)
        ypos = uniform(-2.5, 2.5)
        ysc = uniform(0.9, 1.1)
        xsc = uniform(0.9, 1.1)

        t = mix.ToWorld().scale(x=xsc, y=ysc, z=1).rotate(z=1, angle=zrot).translate(xpos, ypos).get()

        mix.instance(scene_dict, id=cid, ref="chicken", to_world=t)

        cids.append(cid)
    # end

    for cid in cids:
        bbox = bounding_box(scene_dict, cid)

    scene = mix.load_dict(scene_dict)

    print("render ...")
    image = mix.render(scene)

    print("save ...")
    plt.imsave(f"{scene_name}.png", image)

    pass


def main():

    gen_scene()

    pass


if __name__ == "__main__":
    main()
