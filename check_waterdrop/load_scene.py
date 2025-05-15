import mitsuba as mi
from waterdrop import render_water_drop


# mi.set_variant('cuda_ad_mono')
# mi.set_variant('cuda_ad_rgb')
# mi.set_variant("llvm_ad_mono")      # MISSING SUPPORT fma
mi.set_variant("scalar_rgb")


SCENE_NAME = "drop_scene"
# SCENE_NAME = "drop_segmentation"


def main():
    render_water_drop(
        scene_name="drop_scene",
        camera_shift_z=0,
        scene_shift_z=1,

        drop_base=4,
        drop_height=.5,
        drop_shift_xy=0,

        table_rot_z=0,
        # table_shift_z=0,

        dispenser_side=1,
        # dispenser_shift_xy=(.1,-.1),
        # dispenser_shift_z=.5,

        save_info=True
    )

    # render_water_drop(
    #     scene_name="drop_segmentation",
    #     camera_shift_z=0,
    #     scene_shift_z=0,
    #
    #     drop_base=2,
    #     drop_height=1,
    #     drop_shift_xy=(-.5,.5),
    #
    #     table_rot_z=0,
    #     table_shift_z=0,
    #
    #     dispenser_side=1,
    #     # dispenser_shift=(.1,-.1),
    #     # dispenser_shift_z=.5,
    #
    #     cmap=None
    # )
# end


if __name__ == "__main__":
    main()
