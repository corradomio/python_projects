import lightning
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm


pi = np.pi
twopi = 2*np.pi


class NonLinear3D(lightning.LightningModule):

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
            nn.Linear(16, 1),
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


class NonLinear4D(lightning.LightningModule):

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


def plot(abcd):
    plt.scatter(abcd[:, 0], abcd[:, 1], s=0.1)
    plt.show()

    plt.scatter(abcd[:, 2], abcd[:, 3], s=0.1)
    plt.show()


def col(a):
    return a.reshape(-1, 1)


def main():
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)

    x, y = np.meshgrid(x, y)

    ro = np.sqrt(x*x + y*y)
    theta = np.arctan2(x, y)
    xy_rt = np.concatenate([col(x), col(y), col(ro), col(theta)], axis=1)

    plot(xy_rt)

    ro = np.arange(0, 1, 0.01)
    theta = np.arange(0, twopi, 0.01)

    ro, theta = np.meshgrid(ro, theta)

    x = ro*np.cos(theta)
    y = ro*np.sin(theta)
    rt_xy = np.concatenate([col(ro), col(theta), col(x), col(y)], axis=1)

    plot(rt_xy)


if __name__ == "__main__":
    main()
