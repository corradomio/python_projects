import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np

print(mi.variants())
mi.set_variant('cuda_ad_rgb')

# scene = mi.load_file("examples/shape_instance_fractal.xml")
# scene = mi.load_file("examples/dragon/scene_v3.xml")
scene = mi.load_file("test_scene.xml")

image = mi.render(scene)
image: np.ndarray = np.array(image)
image = (image ** (1./2.2)).clip(0,1)

plt.gca().set_aspect('equal')
plt.axis("off")
plt.tight_layout()
plt.imshow(image)
plt.show()

# mi.util.write_bitmap("scene.png", image)
