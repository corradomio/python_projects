import mitsuba as mi
from waterdrop import render_water_drop

# [
# 'scalar_rgb', 'scalar_spectral', 'scalar_spectral_polarized',
# 'llvm_ad_rgb', 'llvm_ad_mono', 'llvm_ad_mono_polarized', 'llvm_ad_spectral', 'llvm_ad_spectral_polarized',
# 'cuda_ad_rgb', 'cuda_ad_mono', 'cuda_ad_mono_polarized', 'cuda_ad_spectral', 'cuda_ad_spectral_polarized'
# ]

# mi.set_variant('cuda_ad_mono')
# mi.set_variant('cuda_ad_rgb')
# mi.set_variant("llvm_ad_mono")      # MISSING SUPPORT fma
# mi.set_variant("scalar_rgb")
mi.set_variant("llvm_ad_mono")


SCENE_NAME = "drop_scene"
# SCENE_NAME = "drop_segmentation"


def main():
    print(mi.variants())

    render_water_drop(
        scene_name="drop_scene",
        camera_shift_z=0,
        scene_shift_z=0,

        drop_base=2,
        drop_height=1,
        drop_shift=(-.5,.5),

        table_rot_z=0,
        table_shift_z=0,

        dispenser_side=1,
        # dispenser_shift=(.1,-.1),
        # dispenser_shift_z=.5,

        save_info=True,
        save_dir="images"
    )

    # render_water_drop(
    #     scene_name="drop_segmentation",
    #     camera_shift_z=0,
    #     scene_shift_z=0,
    #
    #     drop_base=2,
    #     drop_height=1,
    #     drop_shift=(-.5,.5),
    #
    #     table_rot_z=0,
    #     table_shift_z=0,
    #
    #     dispenser_side=1,
    #     # dispenser_shift=(.1,-.1),
    #     # dispenser_shift_z=.5,
    #
    #     cmap=None,
    #     save_dir="images"
    # )
# end


if __name__ == "__main__":
    main()
