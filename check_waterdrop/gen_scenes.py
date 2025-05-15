from waterdrop import render_water_drop
import mitsuba as mi
from random import random, choice


def rrandom(rmin, rmax):
    return rmin + (rmax-rmin)*random()


def render_scenes(**kwargs):
    render_water_drop(scene_name="drop_scene", cmap="gray", save_info=True, **kwargs)
    render_water_drop(scene_name="drop_segmentation", cmap=None, save_info=False, **kwargs)


mi.set_variant("scalar_rgb")

N_IMAGES = 10000

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

        camera_shift_z = rrandom(0, 1)
        scene_shift_z = rrandom(0, 1)

        drop_height = rrandom(0, 2)
        if drop_height < 1:
            drop_base = rrandom(2, 4)
        else:
            drop_base = rrandom(0, 2)

        drop_shift_x = rrandom(-.3, .3)
        drop_shift_y = rrandom(-.3, .3)
        table_rot_z = rrandom(-3, 3)
        dispenser_side = choice([0,1])

        args_list.append(dict(
            scene_id=image_id,

            camera_shift_z=camera_shift_z,
            scene_shift_z=scene_shift_z,

            drop_base=drop_base,
            drop_height=drop_height,
            drop_shift_xy=(drop_shift_x, drop_shift_y),

            table_rot_z=table_rot_z,
            # table_shift_z=0,

            dispenser_side=dispenser_side,
            # dispenser_shift_z=0,
        ))

    for args in args_list:
        print(f"-- {args["scene_id"]} --")
        render_scenes(**args)
        # break

    # Parallel(n_jobs=14)(delayed(render_scenes)(**args) for args in args_list)
# end


if __name__ == "__main__":
    main()

