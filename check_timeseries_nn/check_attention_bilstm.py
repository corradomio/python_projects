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
    output_size = yt.shape[1]   # 24 (batch, seq, data)

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnk.LSTM(input=input_size,
                 units=input_size,
                 bidirectional=True,
                 return_sequence=True), nn.Tanh(),
        # (*, 24, 2*19)
        nnk.SeqSelfAttention(input=2*input_size, units=32),
        # (*, 24, 38)
        nnk.TimeDistributed(
            # (24, 38)
            nnk.Dense(input=38, units=output_size)
            # (24, 24)
        ),
        # (*, 24, 24)
        nnk.Dense(input=24, units=1),
        # (*, 24, 1)
        nnx.Probe("last")
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.0001, monitor="valid_loss")

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
    # Plot history

    # history = smodel.history
    # plt.plot(history[:, 'train_loss'], label='train_loss')
    # plt.plot(history[:, 'valid_loss'], label='valid_loss')
    # plt.legend()
    # plt.show()

    # -------------------------------------------------------------------------------
    # Plot Test prediction

    # ypred = smodel.predict(Xp)
    # if len(ypred.shape) == 3:
    #     y_train = pd.Series(data=yt[:, 0, 0], index=it)
    #     y_test = pd.Series(data=yp[:, 0, 0], index=ip)
    #     y_pred = pd.Series(data=ypred[:, 0, 0], index=ip)
    # else:
    #     n = yp.shape[1]
    #     ip = pd.period_range(ip[0], periods=n)
    #     y_train = pd.Series(data=yt[:, 0], index=it)
    #     y_test = pd.Series(data=yp[0, :], index=ip)
    #     y_pred = pd.Series(data=ypred[0, :], index=ip)
    #
    # plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])
    # plt.show()

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

    # plt.plot(yf_pred['EASY'].to_numpy())
    # plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
