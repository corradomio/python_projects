import numpy as np
import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
import torchx.nn as nnx


def main():
    data = np.zeros((1000, 10), dtype=np.float32)
    data[:, 0] = np.random.rand(1000)

    data[:, 1] = 2*data[:, 0] + 0.1*np.random.randn(1000)
    data[:, 2] = np.log(1+data[:, 0]) + 0.1*np.random.randn(1000)
    data[:, 3] = np.sin(4*np.pi*data[:, 0]) + 0.1*np.random.randn(1000)
    data[:, 4] = np.cos(2*np.pi*data[:, 0]) + 0.1*np.random.randn(1000)
    data[:, 5] = data[:, 3]*data[:, 4] + 0.1*np.random.randn(1000)
    data[:, 6] = data[:, 3] + data[:, 4] + 0.1*np.random.randn(1000)
    data[:, 7] = data[:, 3] - data[:, 4] + 0.1 * np.random.randn(1000)
    data[:, 8] = data[:, 5] + np.sqrt(np.abs(data[:, 4]*data[:, 7])) + 0.1 * np.random.randn(1000)
    data[:, 9] = np.sqrt(np.abs(data[:, 6]*data[:, 7])) + 0.1 * np.random.randn(1000)

    for i in range(1, 4):
        plt.scatter(data[:, 0], data[:, i])
        plt.title(f"data[{i}]")
        plt.show()
    pass

    Xt = data
    yt = data

    rbm = nnx.RestrictedBoltzmannMachine(10, 10)

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 10)
        rbm,
        # (*, (10, 10))
        nnx.Probe("last")
    )

    early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.0001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=10,
        criterion=nnx.RestrictedBoltzmannMachineLoss,
        criterion__model=rbm,
        optimizer=torch.optim.Adam,
        lr=0.005,
        callbacks=[early_stop],
        batch_size=6
    )

    smodel.fit(Xt, yt)

    hid = np.random.rand(1000, 10).astype(np.float32)
    gen = rbm.generate(hid)

    for i in range(1, 4):
        plt.scatter(gen[:, 0], gen[:, i])
        plt.title(f"data[{i}]")
        plt.show()
    pass


if __name__ == "__main__":
    main()
