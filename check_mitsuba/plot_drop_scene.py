import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


print(mi.variants())
mi.set_variant('cuda_ad_mono')


def plot(scene_name):
    # scene_name = "drop_scene_5"
    scene = mi.load_file(f"{scene_name}.xml")
    pprint(scene)

    image = mi.render(scene)
    image: np.ndarray = np.array(image)
    image = (image ** (1./2.2)).clip(0,1)

    plt.gcf().set_size_inches(10,10)

    plt.gca().set_aspect('equal')
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(image, cmap='gray')
    plt.savefig(f"{scene_name}.jpg", dpi=300)
    plt.show()

# mi.util.write_bitmap("scene.png", image)


def main():
    # plot("drop_scene_1")
    # plot("drop_scene_2")
    # plot("drop_scene_3")
    # plot("drop_scene_4")
    plot("drop_scene_5")

if __name__ == "__main__":
    main()

