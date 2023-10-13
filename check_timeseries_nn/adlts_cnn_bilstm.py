import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
from sktime.utils.plotting import plot_series

import torchx.nn as nnx
import torchx.keras.layers as nnk
from loaddata import *


def main():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12
    hidden_size = 150

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnk.Reshape(n_dims=1),
        nnx.Probe("reshape"),
        # (*, 24, 19, 1)

        nnk.TimeDistributed(
            # nnx.Probe("td1.in"),
            # (23, 19, 1)       (batch, channels, seq)
            nnk.Conv1D(input=1, filters=128, kernel_size=4), nn.Tanh(),
            # nnx.Probe("td1.conv1d"),
            # (23, 128, 16)
        ),
        nnx.Probe("td1.conv1d"),
        # (*, 24, 128, 16)
        nnk.TimeDistributed(
            # nnx.Probe("td2.in"),
            # (23, 128, 16)
            nnk.Conv1D(input=128, filters=64, kernel_size=2), nn.Tanh(),
            # nnx.Probe("td2.conv1d"),
            # (23, 64, 15)
        ),
        nnx.Probe("td2.conv1d"),
        # (*, 23, 64, 15)
        nnk.TimeDistributed(
            # nnx.Probe("td3.in"),
            # (23, 64, 15)
            nnk.MaxPooling1D(pool_size=2),
            # nnx.Probe("td3.maxp"),
            # (23, 32, 15)
        ),
        nnx.Probe("td3.maxp"),
        # (*, 24, 32, 15)
        nnk.TimeDistributed(
            # nnx.Probe("td4.in"),
            nnk.GlobalMaxPool1D(),
            # nnx.Probe("td4.gmaxp"),
            # (23, 64)
        ),
        nnx.Probe("td4.gmaxp"),
        # (*, 24, 64)
        nnk.LSTM(input=64, units=input_size, return_sequences=True, bidirectional=True), nn.Tanh(),
        # (*, 24, 2*19)
        nnk.Dense(input=(window_len, input_size, 2), units=(predict_len, output_size)),
        nnx.Probe("last")
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=25, threshold=0.0001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        lr=0.0005,
        callbacks=[early_stop],
        batch_size=6
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
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
