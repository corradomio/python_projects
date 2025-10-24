from random import random
import mitsuba as mi
from commons import *
from joblib import Parallel, delayed

from waterdrop import render_water_drop, waterdrop_info


def rrandom(rmin, rmax):
    return rmin + (rmax-rmin)*random()


# mi.set_variant("cuda_ad_rgb")
# mi.set_variant("scalar_rgb")

N_IMAGES = 10000



def render_scenes(kwargs):
    mi.set_variant("cuda_ad_rgb")
    # mi.set_variant("llvm_ad_mono")
    # mi.set_variant("scalar_rgb")

    print(f"-- {kwargs["scene_id"]} --")
    render_water_drop(scene_name="drop_scene", cmap="gray", save_info=True, **kwargs)
    render_water_drop(scene_name="drop_segmentation", cmap=None, save_info=False, **kwargs)



def main():
    # prepare the parameters
    args_list: list[dict] = []

    for image_id in range(N_IMAGES):

        # camera_shift_z      [0,1]
        # scene_shift_z       [0,1]
        #
        # drop_base           [0, 2]
        # drop_height         [0, 2]
        #
        # drop_shift_x        [-.3,+.3]
        # drop_shift_y        [-.3,+.3]
        #
        # table_rot_z         [-3,3]
        #
        # dispenser_side      {0,1}
        # dispenser_shift_x   no
        # dispenser_shift_y   no
        # dispenser_shift_z   no

        # version 1
        # camera_shift_z = rrandom(0, 1)
        # scene_shift_z = rrandom(0, 1)
        # drop_base = rrandom(0, 2)
        # drop_height = rrandom(0, 2)
        # drop_shift_x = rrandom(-.3, .3)
        # drop_shift_y = rrandom(-.3, .3)
        # table_rot_z = rrandom(-3, 3)
        # dispenser_side = 0  # choice([0,1])

        # version 2
        camera_shift_z = 0
        scene_shift_z = 0
        delta = 0.1

        contact_angle = rrandom(0, 180)
        if contact_angle < 90:
            drop_base = rrandom(2, 4)
            wd_info = waterdrop_info(theta=contact_angle, b=drop_base)
        else:
            drop_radius = rrandom(1 - delta, 1 + delta)
            wd_info = waterdrop_info(theta=contact_angle, R=drop_radius)

        drop_shift_x = 0
        drop_shift_y = 0
        table_rot_z = 0
        dispenser_side = 0

        drop_base = wd_info["b"]
        drop_height = wd_info["h"]

        args_list.append(dict(
            scene_id=image_id,

            camera_shift_z=camera_shift_z,
            scene_shift_z=scene_shift_z,

            drop_base=drop_base,
            drop_height=drop_height,
            drop_shift_xy=(drop_shift_x, drop_shift_y),

            table_rot_z=table_rot_z,
            table_shift_z=-1,

            dispenser_side=dispenser_side,
            # dispenser_shift_z=0,
            images_root=IMAGES_ROOT,
        ))

    # for kwargs in args_list:
    #     render_scenes(kwargs)

    Parallel(n_jobs=14)(delayed(render_scenes)(kwargs) for kwargs in args_list)
# end


if __name__ == "__main__":
    main()

