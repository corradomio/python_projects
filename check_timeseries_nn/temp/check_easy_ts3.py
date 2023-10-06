import logging.config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skorch
import torch
import torch.nn as nn
from sktime.utils.plotting import plot_series

import pandasx as pdx
import pandasx.preprocessing as ppx
import skorchx.callbacks
import torchx.keras.layers as nnk


def load_data():
    df: pd.DataFrame = pdx.read_data(
        "../easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )

    # split train/test
    train, test = pdx.train_test_split(df, test_size=24)
    X_train, y_train, X__test, y__test = pdx.xy_split(train, test, target='EASY')

    # add monthly/quarterly means of the target
    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER, datetime=None,
                             target='EASY',
                             method=None,
                             means=True)
    Xp_train = pe.fit_transform(X_train)
    yp_train = y_train
    Xp__test = pe.transform(X__test.index)
    yp__test = y__test

    at = ppx.LagsArrayTransformer(xlags=16, ylags=0, tlags=1, temporal=True, dtype=np.float32)
    Xt, yt = at.fit_transform(Xp_train, yp_train)
    Xp, yp = at.transform(Xp__test, yp__test)

    ft = ppx.LagsArrayForecaster(xlags=1, ylags=1, tlags=1, temporal=False, dtype=np.float64)
    yf = ft.fit(Xp_train, yp_train).transform(Xp__test)

    # n = len(yp)
    # i = 0
    # while i < n:
    #     Xf = ft.step(i)
    #     yf[0, :] = yp[i]
    #     i = ft.set_forecast(i, yf)
    #     pass

    it = None
    ip = None

    return Xt, yt, Xp, yp, it, ip


def bilstm():
    Xt, yt, Xp, yp, it, ip = load_data()

    input_size = Xt.shape[2]  # (batch, seq, data)
    output_size = yt.shape[2]  # (batch, seq, data)

    # tmodule = nn.Sequential(
    #     nn.LSTM(input_size=input_size,
    #             hidden_size=input_size,
    #             bidirectional=True),
    #     nnx.Select(select=[0]),
    #     nn.Tanh(),
    #     nn.Flatten(),
    #     nn.Linear(in_features=input_size*12*2,
    #               out_features=4*output_size),
    #     nn.Unflatten(1, unflattened_size=(4, output_size))
    # )

    tmodule = nn.Sequential(
        nnk.LSTM(input=input_size,
                 units=input_size,
                 bidirectional=True,
                 return_state=False,
                 return_sequence=True),
        # seq: (128,12,2)
        # flat:(128, 2)
        # nnx.Probe("lstm"),
        # nnx.Select(select=0),
        nn.Tanh(),
        # nnx.Probe("tanh"),
        # (128,24)
        nnk.Dense(input=(input_size, 12, 2), units=(4, output_size)),
        # nnk.Dense(input=(input_size, 2), units=(4, output_size)),
        # nnx.Probe("lin"),
    )

    early_stop = skorchx.callbacks.EarlyStopping(min_epochs=250, patience=10, threshold=0.01)

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=5000,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop]
    )

    smodel.fit(Xt, yt)

    history = smodel.history

    # plt.plot(history[:, 'train_loss'], label='train_loss')
    # plt.plot(history[:, 'valid_loss'], label='valid_loss')
    # plt.legend()
    # plt.show()

    ypred = smodel.predict(Xp)
    y_train = pd.Series(data=yt[:, 0, 0], index=it)
    y_test = pd.Series(data=yp[:, 0, 0], index=ip)
    y_pred = pd.Series(data=ypred[:, 0, 0], index=ip)

    plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])
    plt.show()
# end


def main():
    bilstm()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('../logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
