import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import skorch
import torch.nn as nn
import torchx.nn as nnx
import torch.optim as optim
from skorch.callbacks import EarlyStopping
import numpyx as npx


# MEGA PROBLEMA:
# SE si passano i dati ORDINATI, (x ordinato), NON IMPARA NA CIPPA.
# i dati DEVONO ESSERE MESCOLATI


def main():
    period = 12.
    L = 20
    a1 = .0
    b1 = .075

    a2 = .0
    b2 = .01
    c2 = .0
    d2 = 1/period

    a3 = .0
    b3 = .001
    c3 = .0
    d3 = 3/period

    x: np.ndarray = np.arange(0, L*period, 1., dtype=np.float32).reshape((-1, 1))

    def f(x):
        return (a1 + b1*x) + (a2 + b2*x)*np.sin(c2 + d2*2*np.pi*x) + (a3 + b3*x)*np.sin(c3 + d3*2*np.pi*x)

    plt.plot(f(x))
    plt.show()

    return

    x, = npx.ashuffle(x)
    y = f(x)

    lr = 0.001
    num_units=64

    net = skorch.NeuralNetRegressor(
        module=nn.Sequential(
            nn.Linear(1, num_units),
            # nn.ReLU(inplace=True),
            # nn.Linear(num_units, num_units),
            # nn.ReLU(inplace=True),
            nnx.Snake(2*np.pi),
            nn.Linear(num_units, 1)),
        max_epochs=100,
        batch_size=64,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam,
        lr=lr,
        callbacks=[EarlyStopping(patience=5)],
        device='cuda'
    )

    net.fit(x, y)

    x = np.sort(x, axis=0)
    y = (a1 + b1*x) + (a2 + b2*x)*np.sin(x)
    # y = np.sin(x)
    # parameters = list(model.parameters())
    p = net.predict(x)

    plt.plot(x, y)
    plt.plot(x, p)
    plt.show()

    pass


if __name__ == "__main__":
    main()
