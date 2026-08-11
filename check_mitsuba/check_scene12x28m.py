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

        x = uniform(-11.8,11.8)
        y = uniform(-27.8, 27.8)
        angle = uniform(-180,180)

        sx = uniform(0.9, 1.1)
        sy = uniform(0.9, 1.1)
        sz = uniform(0.9, 1.1)

        # t = mix.ToWorld().scale(value=[0.5,0.5,0.5]).scale(value=[side,side,side]).translate(value=[x,y,0]).get()
        t = mix.ToWorld().scale(value=[sx, sy, sz]).rotate(z=1, angle=angle).translate(value=[x,y,0]).get()

        mix.clone(scene_dict, cid, ref="cube0", to_world=t)

        pass

    return scene_dict



def main():
    # scene_name="cornell-box"
    # scene_name="scene-simple"
    # scene_name="examples/scenes/simple"
    # scene_name="examples/scenes/cbox"
    # scene_name="examples/banner_01/scene"
    scene_name="simple12x28m"

    side = 10

    # scene = mix.load_scene(f"{scene_name}.xml", **params)
    scene_dict = mix.load_scene_dict(f"{scene_name}.xml", side=side/100.)
    scene_dict = add_random_cubes(scene_dict, 6500)

    scene = mix.load_dict(scene_dict)

    image = mix.render(scene)

    plt.imsave(f"{scene_name}_{side}cm-1.png", image)
    pass


if __name__ == "__main__":
    main()
