import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon
import numpy as np

def _to_polygon(x, mean, sdev, color, alpha):
    n = len(x)
    xy = [[0,0]]*(n+n)
    j = 0
    for i in range(0, n, +1):
        xy[j] = [x[i], mean[i]-sdev[i]]
        j += 1
    for i in range(n-1, -1, -1):
        xy[j] = [x[i], mean[i]+sdev[i]]
        j += 1
    xy = np.array(xy)
    return Polygon(xy, color=color, alpha=alpha)


def plot(x, mean, sdev=None, **kwargs):
    p = plt.plot(x, mean, **kwargs)
    color = p[-1].get_color()
    poly = _to_polygon(x, mean, sdev, color, 0.2)
    ax = plt.gca()
    """:type:plt.Axes"""
    ax.add_patch(poly)
    # plt.plot(x, mean - sdev, color=p[-1].get_color())
    # plt.plot(x, mean + sdev, color=p[-1].get_color())
# end