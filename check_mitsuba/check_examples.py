from math import pi, sin, cos, radians
from typing import Optional

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from random import uniform
from stdlib import jsonx

import mitsubax as mix

print(mi.variants())
# mi.set_variant('cuda_ad_mono')
# mi.set_variant("llvm_ad_rgb")
# mi.set_variant("scalar_rgb")
mi.set_variant("cuda_ad_rgb")


def main():
    # scene_name="cornell-box/scene"
    # scene_name="examples/scenes/cbox"
    # scene_name = "examples/scenes/shadow_art"
    # scene_name = "examples/scenes/simple"

    scene_name = "examples/banner_05/scene"

    params = {

    }
    print("load ...")
    scene = mix.load_scene(f"{scene_name}.xml", **params)

    print("render ...")
    image = mix.render(scene)

    print("save ...")
    plt.imsave(f"{scene_name}.png", image)

    pass


if __name__ == "__main__":
    main()
