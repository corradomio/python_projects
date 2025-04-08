import mitsuba as mi
import matplotlib.pyplot as plt


print(mi.variants())
mi.set_variant('cuda_ad_rgb')

# scene = mi.load_file("scenes/cbox.xml")
scene = mi.load_file("examples/sphere.xml")

image = mi.render(scene, spp=128)

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2))
plt.show()

# mi.util.write_bitmap("scene.png", image)
