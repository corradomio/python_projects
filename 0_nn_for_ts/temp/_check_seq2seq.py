import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
import torchx.nn as nnx
from sktime.utils.plotting import plot_series

import torchx.keras.layers as nnk
from loaddata import *


def main():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # (batch, seq, data)        19
    output_size = yt.shape[1]   # (batch, seq, data)        24

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    hidden_size = 150

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnk.LSTM(input=input_size,
                 units=input_size,
                 return_sequences=False), nn.Tanh(),
        # (*, 1, 19)
        nnk.Dense(input=(1, input_size), units=hidden_size), nn.ReLU(),
        # (*, 150)
        nnk.RepeatVector(output_size),
        # (*, 24, 150)
        nnk.LSTM(input=hidden_size,
                 units=input_size,
                 return_sequences=True), nn.Tanh(),
        # (*, 24, 19)

        #
        # PERCHE' e' possibile confrontare (*, 24, 24) con (*, 24, 1) ???
        #
        nnk.TimeDistributed(
            nnk.Dense(input=input_size, units=output_size),
            # nn.Identity()
            # nnk.Dense(input=input_size, units=1),
        ),
        # (*, 24, 24)
        #
        # FORZATO l'output a (*, 24, 1)
        #
        nnx.Probe("output"),
        nnk.Dense(input=(output_size, output_size), units=(output_size, 1))
        # (*, 24, 1)
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=50, threshold=0.001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,

        optimizer=torch.optim.Adam,
        lr=0.0008,                      # BETTER alternative to optimizer__lr
        optimizer__betas=(0.9, 0.999),
        optimizer__amsgrad=True,

        criterion=torch.nn.MSELoss,

        callbacks=[early_stop],
        batch_size=23
    )

    smodel.fit(Xt, yt)

    # -------------------------------------------------------------------------------
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'])
    plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('../logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
