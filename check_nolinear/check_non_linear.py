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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    x = np.arange(-6.28, +6.28, 0.001, dtype=float).reshape(-1, 1)
    y = np.sin(x, dtype=float).reshape(-1, 1)

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
