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

print("torch", torch.__version__)
print("pytorch_lightning", pl.__version__)


class NonLinear(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        # self.l1 = nn.Linear(1, 8)
        # self.l2 = nn.Linear(8, 1)

        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        y = self.layers(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss


def polar_to_cart():
    ro = np.arange(0.001, 1, 0.01)
    theta = np.arange(0, 2*np.pi, 0.01)

    ro, theta = np.meshgrid(ro, theta)
    x = ro*np.cos(theta)
    y = ro*np.sin(theta)

    ro = ro.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    c: np.ndarray = np.sign(x)*np.sign(y)
    # c: {-1, 0, +1}
    c0 = (c < 0).reshape(-1)
    c1 = (c >= 0).reshape(-1)

    rtxyc = np.concatenate([ro, theta, x, y, c], axis=1)

    plt.scatter(ro[c0], theta[c0])
    plt.scatter(ro[c1], theta[c1])
    plt.show()

    plt.scatter(x[c0], y[c0])
    plt.scatter(x[c1], y[c1])
    plt.show()

    rt = rtxyc[:, 0:2]
    xy = rtxyc[:, 2:4]

    torch.set_float32_matmul_precision('medium')
    p2c = NonLinear()
    p2c.fit(rt, xy, max_epochs=30)

    xy_hat = p2c.predict(rt)
    c0 = c0.reshape(-1)
    c1 = c1.reshape(-1)

    plt.scatter(xy_hat[c0, 0], xy_hat[c0, 1])
    plt.scatter(xy_hat[c1, 0], xy_hat[c1, 1])
    plt.show()
    pass


def cart_to_polar():
    x_ = np.arange(-1, 1, 0.01)
    y_ = np.arange(-1, 1, 0.01)

    x, y = np.meshgrid(x_, y_)

    ro = np.sqrt(x*x+y*y)
    theta = np.arctan(x, y)

    c: np.ndarray = np.sign(x) * np.sign(y)
    c0 = (c < 0).reshape(-1)
    c1 = (c >= 0).reshape(-1)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ro = ro.reshape(-1, 1)
    theta = theta.reshape(-1, 1)

    plt.scatter(x[c0, 0], y[c0, 0])
    plt.scatter(x[c1, 0], y[c1, 0])
    plt.title('Source')
    plt.show()

    plt.scatter(theta[c0, 0], ro[c0, 0])
    plt.scatter(theta[c1, 0], ro[c1, 0])
    plt.title('Actual')
    plt.show()

    rtxyc = np.concatenate([ro, theta, x, y, c], axis=1)

    rt = rtxyc[:, 0:2]
    xy = rtxyc[:, 2:4]

    torch.set_float32_matmul_precision('medium')
    p2c = NonLinear()
    p2c.fit(xy, rt, max_epochs=30)

    rt_hat = p2c.predict(xy)

    plt.scatter(rt_hat[c0, 0], rt_hat[c0, 1])
    plt.scatter(rt_hat[c1, 0], rt_hat[c1, 1])
    plt.title('Predicted')
    plt.show()


def non_lin():
    from math import sin, cos

    x = np.arange(-6.28, +6.28, 0.1)
    y = np.arange(-6.28, +6.28, 0.1)
    x, y = np.meshgrid(x, y)
    z = np.cos(x) * np.sin(y)
    # z = npx.surface2d(lambda x, y: cos(x)*sin(y), x, y)

    side = x.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-6.28, +6.28)
    ax.axes.set_ylim3d(-6.28, +6.28)
    ax.axes.set_zlim3d(-1, +1)
    plt.title('Actual')
    plt.show()

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    xy = np.concatenate([x, y], axis=1)

    torch.set_float32_matmul_precision('medium')
    model = NonLinear()

    model.fit(xy, z, batch_size=100, max_epochs=100)

    z_hat = model.predict(xy)

    x = x.reshape(-1, side)
    y = y.reshape(-1, side)
    z_hat = z_hat.reshape(-1, side)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_xlim3d(-6.28, +6.28)
    ax.axes.set_ylim3d(-6.28, +6.28)
    ax.axes.set_zlim3d(-1, +1)
    plt.title('Predicted')
    plt.show()


def main():
    # non_lin()
    # polar_to_cart()
    cart_to_polar()

    pass


if __name__ == "__main__":
    main()
