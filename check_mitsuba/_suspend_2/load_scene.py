from math import pi, sin, cos, radians
from typing import Optional

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

import mitsubax as mix

print(mi.variants())
# mi.set_variant('cuda_ad_mono')
mi.set_variant("scalar_rgb")


def render_scene(scene, fname=None):
    image = mi.render(scene)
    image: np.ndarray = np.array(image)
    image = (image ** (1./2.2)).clip(0,1)

    plt.gcf().set_size_inches(10,10)

    plt.gca().set_aspect('equal')
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(image, cmap='gray')
    if fname is not None:
        plt.savefig(f"{fname}.jpg", dpi=300)
    plt.show()

# cornell-box

# print("xml scene")
# scene = mi.load_file("models/cornell-box/scene_v3.xml")
# pprint(scene)
# render_scene(scene)

# print("dict scene")
# scene_dict = mix.load_scene_dict("models/cornell-box/scene_v3.xml")
# scene = mi.load_dict(scene_dict)
# pprint(scene)
# render_scene(scene)

# drop_scene_5
# print("xml scene")
# scene = mi.load_file("drop_scene_5.xml")
# pprint(scene)
# render_scene(scene)

# print("dict scene")
# scene_dict = mix.load_scene_dict(
#     "drop_scene_5.xml",
#     drop_diameter=1,
#     drop_move_to=0)
# scene = mi.load_dict(scene_dict)
# render_scene(scene)

def radius_move(base: float, angle: Optional[float] = None, dangle: Optional[float] = None) -> tuple[float, float]:
    if dangle is not None:
        angle = radians(dangle)

    # angle respect -X
    angle = pi - angle

    beta = pi/2-angle
    R = base/(2*cos(beta))
    Y = R*sin(beta)
    return R, Y

# scene = mix.load_scene(
#     "drop_scene_5.xml",
#     drop_radius=1,
#     drop_move_to=0)
# render_scene(scene)


# scene = mix.load_scene(
#     "drop_scene_5.xml",
#     drop_radius=1,
#     drop_move_to=-1)
# render_scene(scene)

# R, Y = radius_move(2., dangle=90)
# scene = mix.load_scene(
#     "drop_scene_5.xml",
#     drop_radius=R,
#     drop_move_to=Y-1)
# render_scene(scene)


def render_water_drop(contact_angle):
    R, Y = radius_move(2., dangle=contact_angle)
    scene = mix.load_scene(
        "drop_scene_5.xml",
        drop_radius=R,
        drop_move_to=Y-1)

    scene_name=f"images/water_drop_{contact_angle}.png"
    render_scene(scene, scene_name)

# for contat_angle in range(90, 10, -10):
#     render_water_drop(contat_angle)

render_water_drop(10)
# render_water_drop(5)
# render_water_drop(2)
# render_water_drop(1)
