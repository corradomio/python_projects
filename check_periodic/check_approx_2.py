from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import numpyx as npx
import skorch
import skorchx
import torch
import torch.nn as nn
import torchx.nn as nnx
from matplotlib import pyplot as plt


def main():
    # define the dataset
    x = np.asarray([i for i in range(-50, 51)], dtype=np.float32)
    y = np.asarray([i ** 2.0 for i in x], dtype=np.float32)
    print(x.min(), x.max(), y.min(), y.max())

    # reshape arrays into into rows and cols
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))

    # separately scale the input and output variables
    scale_x = MinMaxScaler()
    x = scale_x.fit_transform(x)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)
    print(x.min(), x.max(), y.min(), y.max())

    n = 10

    AF = nn.LeakyReLU

    early_stop = skorch.callbacks.EarlyStopping(patience=100, threshold=0, monitor="valid_loss")
    print_log = skorchx.callbacks.PrintLog(delay=3)

    module = nn.Sequential(
        nnx.Linear(in_features=1, out_features=n),
        AF(),
        nnx.Linear(in_features=n, out_features=n),
        AF(),

        # nnx.Linear(in_features=n, out_features=2*n),
        # nn.ReLU(),
        # nnx.Linear(in_features=2*n, out_features=n),
        # nn.ReLU(),

        nnx.Linear(in_features=n, out_features=1)
    )

    model = skorch.NeuralNetRegressor(
        module=module,
        max_epochs=1000,
        callbacks=[early_stop],
        callbacks__print_log=print_log,
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.RMSprop,
        # optimizer=torch.optim.Adagrad,
        # lr=0.001,
        batch_size=10
    )

    model.fit(x, y)

    # make predictions for the input data
    yhat = model.predict(x)

    # inverse transforms
    x_plot = scale_x.inverse_transform(x)
    y_plot = scale_y.inverse_transform(y)
    yhat_plot = scale_y.inverse_transform(yhat)

    # report model error
    print('MSE: %.3f' % mean_squared_error(y_plot, yhat_plot))

    # plot x vs y
    plt.scatter(x_plot, y_plot, label='Actual')

    # plot x vs yhat
    plt.scatter(x_plot, yhat_plot, label='Predicted')
    plt.title('Input (x) versus Output (y)')
    plt.xlabel('Input Variable (x)')
    plt.ylabel('Output Variable (y)')
    plt.legend()
    plt.show()

    pass


if __name__ == "__main__":
    main()
