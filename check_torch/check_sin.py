import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from skorch.regressor import NeuralNetRegressor


class SinModule(nn.Module):
    def __init__(self, num_units=20):
        super(SinModule, self).__init__()
        self.lin1 = nn.Linear(1, num_units, dtype=torch.float)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(num_units, num_units, dtype=torch.float)
        self.act2 = nn.Tanh()
        self.lin3 = nn.Linear(num_units, 1, dtype=torch.float)
        self.act3 = nn.Tanh()


    def forward(self, X, **kwargs):
        T = self.lin1(X)
        T = self.act1(T)
        T = self.lin2(T)
        T = self.act2(T)
        T = self.lin3(T)
        T = self.act3(T)
        return T


def main():
    s=0.01
    X_train = np.arange(0,4*np.pi+s/2, s, dtype=np.float32)
    y_train = np.sin(X_train)

    plt.plot(X_train, y_train)
    plt.title("Train data")
    plt.show()

    net = NeuralNetRegressor(
        SinModule,
        max_epochs=1000,
        lr=0.001,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        device="cuda"
    )
    net.fit(X_train.reshape((-1,1)), y_train.reshape((-1,1)))

    X_test = np.arange(1,3, 0.1, dtype=np.float32)
    y_test = np.sin(X_test)
    y_pred = net.predict(X_test.reshape((-1,1)))[:, 0]

    plt.clf()
    plt.plot(X_test, y_test, c="b")
    plt.plot(X_test, y_pred, c="g")
    plt.title("Test/predicted data")
    plt.show()

    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Trained model MSE={results}.")


if __name__ == "__main__":
    main()