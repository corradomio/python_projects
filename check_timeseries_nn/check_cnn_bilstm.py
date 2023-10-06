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

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    output_size = yt.shape[1]   # 24 (batch, seq, data)

    # -------------------------------------------------------------------------------
    # Conv1d
    #       in_channels: int,
    #       out_channels: int,
    #       kernel_size: _size_1_t,
    #       stride: _size_1_t = 1,
    #       padding: Union[str, _size_1_t] = 0,
    #       dilation: _size_1_t = 1,
    #       groups: int = 1,
    #       bias: bool = True,
    #       padding_mode: str = 'zeros',  # TODO: refine this type
    #       device=None,
    #       dtype=None

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.ReshapeVector(n_dims=1),
        # (*, 24, 19, 1)
        nnx.TimeDistributed(
            # (24, 19, 1)
            nnk.Conv1D(input=1, filters=128, kernel_size=4, channels_last=True), nn.Tanh(),
            # (24, 16, 128)
        ),
        # (*, 24, 16, 128)
        nnx.TimeDistributed(
            # (24, 16, 128)
            nnk.Conv1D(input=128, filters=64, kernel_size=2, channels_last=True), nn.Tanh(),
            # (24, 15, 64)
        ),
        # (*, 24, 15, 64)
        nnk.TimeDistributed(
            # (24, 15, 64)
            nnk.MaxPooling1D(pool_size=2)
            # (24, 7, 64)
        ),
        # (*, 24, 7, 64)
        nnk.TimeDistributed(
            # (24, 7, 64)
            nnk.GlobalMaxPool1D()
            # (24, 64)
        ),
        # (*, 24, 64)
        nnk.LSTM(input=64, units=input_size, bidirectional=True, return_sequence=True), nn.Tanh(),
        # (*, 24, 19*2)
        nnk.TimeDistributed(
            # (24, 38)
            nnk.Dense(input=38, units=output_size)
            # (24, 24)
        ),
        # (*, 24, 24)
        nnk.Dense(input=output_size, units=1),
        # (*, 24, 1)
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0.001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop],
        batch_size=128
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
