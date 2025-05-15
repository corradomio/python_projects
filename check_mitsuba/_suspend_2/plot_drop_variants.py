from math import pi, sin, cos, radians
from typing import Optional

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from pprint import pprint
import mitsubax as mix

print(mi.variants())
mi.set_variant('cuda_ad_mono')
# mi.set_variant("scalar_rgb")


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
# end


def radius_move(base: float, angle: Optional[float] = None, dangle: Optional[float] = None) -> tuple[float, float]:
    if dangle is not None:
        angle = radians(dangle)

    # angle respect -X
    angle = pi - angle

    beta = pi/2-angle
    R = base/(2*cos(beta))
    Y = R*sin(beta)
    return R, Y


def render_water_drop(contact_angle):
    R, Y = radius_move(2., dangle=contact_angle)
    scene = mix.load_scene(
        "drop_scene_5.xml",
        drop_radius=R,
        drop_move_to=Y-1)

    scene_name=f"images/water_drop_{contact_angle}.png"
    render_scene(scene, scene_name)

render_water_drop(10)
