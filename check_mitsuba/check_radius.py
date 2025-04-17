import numpy as np
import mitsuba as mi
import mitsubax as mix
import matplotlib.pyplot as plt


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

scene = mix.load_scene(
    "drop_scene_v1.xml",
    drop_radius=[1,1,.75],
    drop_move_to=0)


render_scene(scene)
