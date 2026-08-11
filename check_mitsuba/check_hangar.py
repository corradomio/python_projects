from random import uniform

import matplotlib.pyplot as plt
import mitsuba as mi

import mitsubax as mix

print("Mitsuba", mi.__version__)
print(mi.variants())
# mi.set_variant('cuda_ad_mono')
# mi.set_variant("scalar_rgb")
mi.set_variant("cuda_ad_rgb")


def add_random_cubes(scene_dict: dict, n: int):

    for i in range(n):
        cid = f"c@{i+1}"

        x = uniform(-6,6)
        y = uniform(-40, 40)

        # t = mix.ToWorld().scale(value=[0.5,0.5,0.5]).scale(value=[side,side,side]).translate(value=[x,y,0]).get()
        t = mix.ToWorld().translate(value=[x,y,0]).get()

        mix.instance(scene_dict, id=cid, ref="cube0", to_world=t)

        pass

    return scene_dict



def main():
    # scene_name="cornell-box"
    # scene_name="scene-simple"
    # scene_name="examples/scenes/simple"
    # scene_name="examples/scenes/cbox"
    # scene_name="examples/banner_01/scene"
    scene_name="hangar"

    side=0.10

    # scene = mix.load_scene(f"{scene_name}.xml", **params)
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml", side=side)
    scene_dict = add_random_cubes(scene_dict, 20000)

    scene = mix.load_dict(scene_dict)
    image = mix.render(scene)

    plt.imsave(f"{scene_name}.png", image)
    pass


if __name__ == "__main__":
    main()
