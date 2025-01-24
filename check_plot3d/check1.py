import matplotlib.pyplot as plt
import matplotlibx.pyplot as pltx
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)


# Plot the surface.

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# ------------------------------------------------------------------

fig, ax, surf = pltx.plot3d(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter('{x:.02f}')
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
