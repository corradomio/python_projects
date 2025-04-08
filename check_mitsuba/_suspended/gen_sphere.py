import numpy as np
import pandas as pd

theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
r = 1

# Convert to Cartesian coordinates
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
print(f"Shape of x, y, z: {x.shape}")
print(f"Min and max values - x: ({x.min():.2f}, {x.max():.2f}), y: ({y.min():.2f}, {y.max():.2f}), z: ({z.min():.2f}, {z.max():.2f})")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Sphere')
ax.set_box_aspect((1, 1, 1))
plt.show()

