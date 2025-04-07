import mitsuba as mi
print(mi.variants())
mi.set_variant('cuda_ad_rgb')

scene = mi.load_file("scenes/simple.xml")

original_image = mi.render(scene, spp=128)

import matplotlib.pyplot as plt
plt.axis('off')
plt.imshow(original_image ** (1.0 / 2.2));
plt.show()
