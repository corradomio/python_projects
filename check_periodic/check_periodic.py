import numpy as np
import skorch
import skorchx
import torch
import torch.nn as nn
import torchx.nn as nnx
from matplotlib import pyplot as plt


def main():
    x = np.arange(100, dtype=np.float32)
    y = np.sin(2*np.pi*x/48).astype(np.float32)
    y = np.power((x-33)/25, 2)
    # y = 1 + 0.1*x

    plt.plot(x, y)
    plt.show()

    x_train = x[:80]
    x_test = x[80:]

    y_train = y[:80]
    y_test = y[80:]

    n = 32

    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0, monitor="valid_loss")
    print_log = skorchx.callbacks.PrintLog(delay=3)

    module = nn.Sequential(
        nnx.Linear(in_features=1, out_features=n),
        nn.ReLU(),
        # nnx.Linear(in_features=n, out_features=2*n),
        # nn.ReLU(),
        # nnx.Linear(in_features=2*n, out_features=n),
        # nn.ReLU(),
        nnx.Linear(in_features=n, out_features=1)
    )

    net = skorch.NeuralNetRegressor(
        module=module,
        max_epochs=100000,
        callbacks=[early_stop],
        callbacks__print_log=print_log,
        # optimizer=torch.optim.RMSprop,
        optimizer=torch.optim.Adagrad,
        lr=0.0002
    )

    net.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred = net.predict(x_test.reshape(-1, 1)).reshape(-1)

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()

    y_pred = net.predict(x_train.reshape(-1, 1)).reshape(-1)

    plt.plot(y_train)
    plt.plot(y_pred)
    plt.show()

    pass


if __name__ == "__main__":
    main()
