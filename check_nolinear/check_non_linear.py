from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning
import pytorch_lightning as pl
import matplotlib.pyplot as plt

print("torch", torch.__version__)
print("pytorch_lightning", pl.__version__)


class PowerModule(nn.Module):

    def __init__(self, order: Union[int, float, list] = 1, cross: int = 1):
        super().__init__()
        if isinstance(order, float):
            self.order = [1, order]
        elif isinstance(order, int):
            self.order = list(range(1, order+1))
        elif isinstance(order, (list, tuple)):
            self.order = order
        else:
            raise ValueError(f"Unsupported order '{order}'")
        self.cross = cross

    def forward(self, x):
        xcat = []
        for e in self.order:
            xe = torch.pow(x, e)
            xcat.append(xe)
        return torch.cat(xcat, dim=1)


class NonLinear(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        # self.l1 = nn.Linear(1, 8)
        # self.l2 = nn.Linear(8, 1)

        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),

            # nn.Linear(1, 32),
            # nn.ReLU(),
            # nn.Linear(32, 1),

            # nn.Linear(1, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1)
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


def main():
    x = np.arange(-6.28, +6.28, 0.001, dtype=float).reshape(-1, 1)
    y = np.sin(x, dtype=float).reshape(-1, 1)

    X = torch.tensor([[1, 11], [2, 22], [3, 33]])
    pm = PowerModule(order=[1, 0.5, 2])
    Y = pm(X)
    print(Y)


    plt.scatter(x, y, s=0.1)
    plt.show()

    torch.set_float32_matmul_precision('medium')
    model = NonLinear()

    model.fit(x, y, batch_size=100, max_epochs=100)

    y_hat = model.predict(x)

    plt.clf()
    plt.scatter(x, y, s=0.1)
    plt.scatter(x, y_hat, s=0.1)
    plt.show()

    pass


if __name__ == "__main__":
    main()
