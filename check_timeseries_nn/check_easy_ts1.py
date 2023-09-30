import logging.config

import numpy as np
import matplotlib.pyplot as plt
import pandasx as pdx
import pandasx.preprocessing as ppx
import pandas as pd
import torch
import torch.nn as nn
import skorch
import skorchx.callbacks
import sktimex as sktx
import torchx.keras.layers as nnk
import torchx.nn as nnx
from numpyx.transformers import MinMaxScaler
from sktime.utils.plotting import plot_series


def load_data():
    N = 24
    df: pd.DataFrame = pdx.read_data(
        "easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )

    TARGET = ['EASY']

    # pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH, datetime=('DATE', 'M'), target='EASY')
    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER, datetime=None, target=TARGET,
                             periods=False)
    dfx = pe.fit_transform(df)

    lt = ppx.LagsTransformer(target='EASY', lags=[1, 16])
    dfl = lt.fit_transform(dfx)

    ss = pdx.preprocessing.StandardScaler(columns=None)
    dfs = ss.fit_transform(dfl)

    ix = df.index

    y = df['EASY'].to_numpy(dtype=np.float32).reshape((-1, 1))
    scaler = MinMaxScaler(bycolumns=True)
    y_scaled = scaler.fit_transform(y)
    y_orig = scaler.inverse_transform(y_scaled)

    tt = sktx.RNNTrainTransform3D(lags=[1, 12], tlags=[0, 1, 2, 3])
    Xt, yt = tt.fit_transform(None, y_scaled)
    n = len(Xt)
    ix = ix[:n]

    Xp = Xt[-N:]
    yp = yt[-N:]
    ip = ix[-N:]

    Xt = Xt[:-N]
    yt = yt[:-N]
    it = ix[:-N]

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
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
