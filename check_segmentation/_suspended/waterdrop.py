import os

import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi
import mitsubac as mix
import time

from stdlib.jsonx import dump
from math import pi, sin, atan2, degrees
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# ROOT_DIR = "E:/Datasets/WaterDrop/orig/"
ROOT_DIR = "images"

twopi = pi*2
halfpi = pi/2


def sq(x): return x*x


# ---------------------------------------------------------------------------
# Droplet
# ---------------------------------------------------------------------------

def droplet_radius_angle_zmove(B: float, H: float) -> tuple[float, float, float, float]:
    """
    Compute R, beta, Z from B and H, where

    - R radius of the drop
    - beta angle of contact respect the drop center
    - Z vertical shift of the drop center to have the intersection with the table
      (y = 0) with length B

    Note: the contact_angle 'alpha' (for a sphere) is

        alpha = pi/2-beta

    :param B: length of the drop base (maximum: 2R)
    :param H: height of the drop respects the table (maximum 2R)
    :return: R (drop radius),
             beta: angle of the contact point respects the drop center
             alpha: contact angle
             Z (down shift of the drop to have the intersection with the table with length B)
    """
    b4h = sq(B)+ 4*sq(H)
    R = b4h/(8*H)
    beta = atan2((sq(B) - 4*sq(H))/b4h, 4*B*H/b4h)
    alpha = halfpi - beta
    Z = R*sin(beta)

    return R, beta, alpha, -Z
# end


def render_scene(scene, fname: Optional[str] = None, cmap:Optional[str]="gray"):
    rimage = mi.render(scene)

    image: np.ndarray = np.array(rimage)
    image = (image ** (1./2.2)).clip(0,1)

    plt.gcf().set_size_inches(6.4,4.8)

    plt.gca().set_axis_off()
    plt.gca().set_aspect('equal')
    plt.imshow(image, cmap=cmap)

    plt.tight_layout()

    if fname is not None:
        # plt.savefig(f"{fname}", dpi=100)
        mi.util.write_bitmap(f"{fname}", rimage)
    # plt.show()
# end


def render_water_drop(
        scene_name: str,
        drop_base: float,
        drop_height: float,
        camera_shift_z: float=0,
        scene_shift_z: float=0,
        drop_shift: Union[float, list[float], tuple[float, float]]=0,
        table_rot_z: float=0,
        table_shift_z: float=0,
        dispenser_side: float=0.25,
        dispenser_shift: Union[float, list[float], tuple[float, float]]=0,
        dispenser_shift_z: float=0,
        scene_id=int(time.time()),
        cmap="gray",
        save_info=False,
        save_dir=None
):
    drop_shift_x, drop_shift_y = \
        (drop_shift, drop_shift) if isinstance(drop_shift, (int, float)) else drop_shift
    dispenser_shift_x, dispenser_shift_y = \
        (dispenser_shift, dispenser_shift) if isinstance(dispenser_shift, (int, float)) else dispenser_shift

    # beta is in degree
    R, beta, alpha, Z = droplet_radius_angle_zmove(drop_base, drop_height)
    # contact_angle = 90 - drop_angle
    drop_angle = beta*180/pi
    contact_angle = alpha*180/pi

    params = dict(
        scene_id=scene_id,
        drop_base=drop_base,
        drop_height=drop_height,

        contact_angle=contact_angle,
        drop_angle=beta,

        camera_shift_z=camera_shift_z,
        scene_shift_z=scene_shift_z,

        drop_radius=R,
        drop_shift_x=drop_shift_x,
        drop_shift_y=drop_shift_y,
        drop_shift_z=Z,

        table_rot_z=table_rot_z,
        table_shift_z=table_shift_z,

        dispenser_side=dispenser_side,
        dispenser_shift_x=dispenser_shift_x,
        dispenser_shift_y=dispenser_shift_y,
        dispenser_shift_z=dispenser_shift_z
    )

    scene = mix.load_scene(f"{scene_name}.xml", **params)

    root_dir = ROOT_DIR if save_dir is None else save_dir

    image_dir = f"{root_dir}/{scene_id//1000:04}"
    os.makedirs(image_dir, exist_ok=True)

    image_name=f"{image_dir}/{scene_name}-{scene_id:04}.png"
    render_scene(scene, image_name, cmap=cmap)

    if save_info:
        param_name = f"{image_dir}/{scene_name}-{scene_id:04}.json"
        dump(params, param_name)
# end
