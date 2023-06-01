import matplotlib.pyplot as plt
import numpy as np
import skorch
import torch.nn as nn
import torchx.nn as nnx
import torch.optim as optim
from skorch.callbacks import EarlyStopping

# MEGA PROBLEMA:
# SE si passano i dati ORDINATI, (x ordinato), NON IMPARA NA CIPPA.
# i dati DEVONO ESSERE MESCOLATI


# class LinApprox(nn.Module):
#     def __init__(self, num_units=10):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(in_features=1, out_features=num_units),
#             # nn.ReLU(),
#             # nn.Linear(in_features=num_units, out_features=2*num_units),
#             # nn.ReLU(),
#             # nn.Linear(in_features=2*num_units, out_features=num_units),
#             nn.ReLU(),
#             nn.Linear(in_features=num_units, out_features=1)
#         )
#     # end
#
#     def forward(self, x):
#         y = self.seq(x)
#         return y

class LinApprox(nn.Module):
    def __init__(self, num_units=10):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_units),
            # nn.ReLU(inplace=True),
            # nn.Linear(num_units, num_units),
            # nn.ReLU(inplace=True),
            nnx.Snake(2 * np.pi),
            nn.Linear(num_units, 1))

    def forward(self, x):
        output = self.regressor(x)
        return output


def main():
    f = 2 * np.pi
    a1 = 0
    b1 = 1
    a2 = .3
    b2 = .5
    x: np.ndarray = np.arange(0, f*5, .001, dtype=np.float32).reshape((-1, 1))
    np.random.shuffle(x)
    y = (a1 + b1 * x) + (a2 + b2 * x) * np.sin(x)
    # y = np.sin(x)

    # plt.plot(x, y)
    # plt.show()

    model = LinApprox(num_units=512)
    lr = 0.001

    net = skorch.NeuralNetRegressor(
        module=model,
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
    y = (a1 + b1 * x) + (a2 + b2 * x) * np.sin(x)
    # y = np.sin(x)
    # parameters = list(model.parameters())
    p = net.predict(x)

    plt.plot(x, y)
    plt.plot(x, p)
    plt.show()

    pass


if __name__ == "__main__":
    main()
