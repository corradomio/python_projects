import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Surface = Poly3DCollection


def plot3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, norm=None, vmin=None,
           vmax=None, lightsource=None, axlim_clip=False, **kwargs) -> tuple[Figure, Axes, Surface]:

    if len(X.shape) == 1:
        X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z,
                    norm=norm, vmin=vmin, vmax=vmax, lightsource=lightsource, axlim_clip=axlim_clip, **kwargs)

    return fig, ax, surf


