from typing import Union
import numpy as np
import numpyx as npx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pi = np.pi
twopi = 2*np.pi


def ro():
    x = np.arange(-10, +10, 0.1)
    y = np.arange(-10, +10, 0.1)
    x, y = np.meshgrid(x, y)
    z = np.sqrt(np.power(x, 2) + np.power(y, 2))
    # z = npx.surface2d(lambda x, y: cos(x)*sin(y), x, y)

    side = x.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(0, 14)
    plt.title('Actual')
    plt.show()

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    xy = np.concatenate([x, y], axis=1)

    torch.set_float32_matmul_precision('medium')
    model = NonLinear3D()

    model.fit(xy, z, batch_size=100, max_epochs=100)

    z_hat = model.predict(xy)

    x = x.reshape(-1, side)
    y = y.reshape(-1, side)
    z_hat = z_hat.reshape(-1, side)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(0, 14)
    plt.title('Predicted')
    plt.show()
    pass


def theta():
    x = np.arange(-10, +10, 0.1)
    y = np.arange(-10, +10, 0.1)
    x, y = np.meshgrid(x, y)
    z = np.arctan2(x, y)

    side = x.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(-2.5, 2.5)
    plt.title('Actual')
    plt.show()

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    xy = np.concatenate([x, y], axis=1)

    torch.set_float32_matmul_precision('medium')
    model = NonLinear3D()

    model.fit(xy, z, batch_size=100, max_epochs=100)

    z_hat = model.predict(xy)

    x = x.reshape(-1, side)
    y = y.reshape(-1, side)
    z_hat = z_hat.reshape(-1, side)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(-2.5, 2.5)
    plt.title('Predicted')
    plt.show()
    pass


def ro_theta():
    x = np.arange(-10, +10, 0.1)
    y = np.arange(-10, +10, 0.1)
    x, y = np.meshgrid(x, y)

    side = x.shape[1]

    ro = np.sqrt(np.power(x, 2) + np.power(y, 2))
    theta = np.arctan2(x, y)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ro = ro.reshape(-1, 1)
    theta = theta.reshape(-1, 1)

    xy = np.concatenate([x, y], axis=1)
    rt = np.concatenate([ro, theta], axis=1)

    torch.set_float32_matmul_precision('medium')
    model = NonLinear4D()

    model.fit(xy, rt, batch_size=100, max_epochs=100)

    rt_hat = model.predict(xy)

    x = x.reshape(-1, side)
    y = y.reshape(-1, side)
    ro_hat = rt_hat[:, 0].reshape(-1, side)
    theta_hat = rt_hat[:, 1].reshape(-1, side)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, ro_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(0, 14)
    plt.title('Ro Predicted')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, theta_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-10, +10)
    ax.axes.set_ylim3d(-10, +10)
    ax.axes.set_zlim3d(-2.5, 2.5)
    plt.title('Theta Predicted')
    plt.show()
    pass


def xy():
    twopi = 2*pi

    ro = np.arange(0., 14.1421, 0.1)
    theta = np.arange(0, twopi, 0.1)
    ro, theta = np.meshgrid(ro, theta)

    n, m = ro.shape
    side = ro.shape[1]

    x = ro*np.cos(theta)
    y = ro*np.sin(theta)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ro = ro.reshape(-1, 1)
    theta = theta.reshape(-1, 1)

    xy = np.concatenate([x, y], axis=1)
    rt = np.concatenate([ro, theta], axis=1)

    torch.set_float32_matmul_precision('medium')
    model = NonLinear4D()

    model.fit(rt, xy, batch_size=100, max_epochs=100)

    xy_hat = model.predict(rt)

    ro = ro.reshape(-1, side)
    theta = theta.reshape(-1, side)
    x_hat = xy_hat[:, 0].reshape(-1, side)
    y_hat = xy_hat[:, 1].reshape(-1, side)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(ro, theta, x_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(0, 14.1421)
    ax.axes.set_ylim3d(0, twopi)
    ax.axes.set_zlim3d(-10, 10)
    plt.title('X Predicted')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(ro, theta, y_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(0, 14.1421)
    ax.axes.set_ylim3d(0, twopi)
    ax.axes.set_zlim3d(-10, 10)
    plt.title('Y Predicted')
    plt.show()
    pass
