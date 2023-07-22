import matplotlib.pyplot as plt

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


def locations(x, y, closed=False, scatter=dict(), arrow=dict()):
    # plt.scatter(x[1:], y[1:], **scatter)
    # scatter["s"] = 10*scatter["s"]
    # scatter["c"] = "red"
    # plt.scatter(x[0:1], y[0:1], **scatter)
    plt.scatter(x, y, **scatter)
    # del scatter["s"]
    # plt.plot(x, y, **scatter)
# end
def locations_start(x, y, closed=False, scatter=dict(), arrow=dict()):
    scatter["c"] = "red"
    plt.scatter(x[0:1], y[0:1], **scatter)
# end


def arrows(x, y, closed=False, scatter=dict(), arrow=dict()):
    plt.scatter(x[1:], y[1:], **scatter)
    scatter["c"] = "red"
    plt.scatter(x[0:1], y[0:1], **scatter)
    n = len(x)
    for i in range(0,n-1):
        j = i+1
        plt.arrow(x[i],y[i], x[j]-x[i],y[j]-y[i], **arrow)
    if closed:
        plt.arrow(x[i], y[i], x[i] - x[0], y[i] - y[0], **arrow)

    # for i in range(0,n-1):
    #     plt.text(x[i], y[i], str(i))
# end
