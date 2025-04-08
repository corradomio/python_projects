import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np

mi.set_variant('scalar_rgb')

image = mi.render(mi.load_dict(mi.cornell_box()))
image: np.ndarray = np.array(image)
image = (image ** (1./2.2)).clip(0,1)

print(image.min())
print(image.max())

plt.axis("off")
plt.imshow(image)
plt.show()

# mi.Bitmap(img).write('cbox.exr')
# mi.util.write_bitmap("cbox.png", img)
