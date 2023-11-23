import logging.config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skorch
import torch
import torch.nn as nn
import torchx.nn as nnx
from sktime.utils.plotting import plot_series

import pandasx as pdx
import pandasx.preprocessing as ppx
import skorchx.callbacks
import torchx.keras.layers as nnk


def lags_transform(X, y):
    at = ppx.LagsArrayTransformer(xlags=[0, 1], ylags=[], tlags=[0, 1],
                                  temporal=False,
                                  channels=False,
                                  y_flatten=True,
                                  dtype=np.float32)
    Xt, yt, it = at.fit(X, y).transform(X, y)
    pass


def load_data():
    df: pd.DataFrame = pdx.read_data(
        "../easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )

    train, test_ = pdx.train_test_split(df, test_size=24)

    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             datetime=None,
                             target='EASY',
                             method=None,
                             means=True)

    train_p = pe.fit_transform(train)
    test__p = pe.transform(test_)

    lt = ppx.LagsTransformer(xlags=[0], ylags=range(17), target='EASY')
    train_l = lt.fit_transform(train_p)
    test__l = lt.transform(test__p)

    X_train, y_train, X_test_, y_test_ = pdx.xy_split(train_l, test__l, target='EASY', shared='EASY')

    scx = ppx.StandardScaler()
    Xs_train = scx.fit_transform(X_train)
    Xs_test_ = scx.transform(X_test_)

    scy = ppx.StandardScaler()
    ys_train = scy.fit_transform(y_train)
    ys_test_ = scy.transform(y_test_)
    # Xs_train, ys_train, Xs_test_, ys_test_ = X_train, y_train, X_test_, y_test_

    lags_transform(Xs_train, ys_train)

    at = ppx.LagsArrayTransformer(xlags=24, ylags=0, tlags=24,
                                  temporal=True,
                                  y_flatten=False,
                                  dtype=np.float32)
    Xt, yt, it = at.fit(Xs_train, ys_train).transform(Xs_train, ys_train)
    # Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    return Xt, yt, it, ys_train, Xs_test_, ys_test_, at


def bilstm():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()

    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # (batch, seq, data)
    output_size = yt.shape[1]   # (batch, seq, data)

    tmodule = nn.Sequential(
        nnk.LSTM(input=input_size,
                 units=input_size,
                 bidirectional=True,
                 return_state=False,
                 return_sequences=False),
        # nnx.Probe("lstm"),
        nn.Tanh(),
        # nn.ReLU(),
        # nnx.Probe("tanh"),
        nnk.Dense(input=(input_size, 2), units=(output_size, 1)),
        # nnx.Probe("dense"),
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0.001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop],
        batch_size=32
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

# end


def main():
    bilstm()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('../logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
