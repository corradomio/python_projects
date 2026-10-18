from random import uniform

import matplotlib.pyplot as plt
import mitsuba as mi

import mitsubax as mix
import mitsubax.procedural_shapes as mixps

print("Mitsuba", mi.__version__)
print(mi.variants())
# mi.set_variant('cuda_ad_mono')
# mi.set_variant("scalar_rgb")
mi.set_variant("cuda_ad_rgb")


def main():
    # scene_name="cornell-box"
    # scene_name="scene-simple"
    # scene_name="examples/scenes/simple"
    # scene_name="examples/scenes/cbox"
    # scene_name="examples/banner_01/scene"
    scene_name="hen_cage"

    scene_dict = mix.load_scene_dict(f"{scene_name}.xml")

    scene = mix.load_dict(scene_dict)
    image = mix.render(scene)

    plt.imsave(f"{scene_name}.png", image)
    pass


if __name__ == "__main__":
    main()
