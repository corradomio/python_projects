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

    ss = ppx.StandardScaler()
    dfs = ss.fit_transform(df)
    dfo = ss.inverse_transform(dfs)

    TARGET = ['EASY']

    PAST = 16
    FUTURE = 24

    pe = ppx.PeriodicEncoder(TARGET, periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER)
    dfx = pe.fit_transform(df)

    dfp = dfx.iloc[:-FUTURE]    # past
    dff = dfx.iloc[-FUTURE:]    # future

    lt = ppx.LagsTransformer(target=TARGET, xlags=1, ylags=16)
    X = lt.fit_transform(dfp)

    at = ppx.ArrayTransformer(target=TARGET, xlags=16, ylags=1)
    Xt, yt = at.fit_transform(X)

    # y = dfs[TARGET].to_numpy(dtype=np.float32)
    # X = dfs[dfs.columns.difference(TARGET)].to_numpy(dtype=np.float32)
    # ix = dfs.index

    # tt = sktx.RNNTrainTransform3D(lags=[12, 12], tlags=[0, 1, 2, 3])
    # Xt, yt = tt.fit_transform(X, y)

    n = len(Xt)
    ix = Xt.index

    Xp = Xt[-N:]
    yp = yt[-N:]
    ip = ix[-N:]

    Xt = Xt[:-N]
    yt = yt[:-N]
    it = ix[:-N]

    return Xt, yt, Xp, yp, it, ip


def plots(smodel, yt, yp, ypred, it, ip):
    history = smodel.history
    plt.plot(history[:, 'train_loss'], label='train_loss')
    plt.plot(history[:, 'valid_loss'], label='valid_loss')
    plt.legend()
    plt.show()

    y_train = pd.Series(data=yt[:, 0, 0], index=it)
    y_test = pd.Series(data=yp[:, 0, 0], index=ip)
    y_pred = pd.Series(data=ypred[:, 0, 0], index=ip)

    plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])
    plt.show()


def bilstm():
    Xt, yt, Xp, yp, it, ip = load_data()

    seq_len = Xt.shape[1]
    input_size = Xt.shape[2]  # (batch, seq, data)
    output_size = yt.shape[2]  # (batch, seq, data)

    tmodule = nn.Sequential(
        nnk.LSTM(input=input_size,
                 units=input_size,
                 bidirectional=True,
                 return_state=False,
                 return_sequence=True,
                 activation='tanh'),
        # nn.Tanh(),
        nnk.Dense(input=(input_size, seq_len, 2), units=(4, output_size)),
    )

    early_stop = skorchx.callbacks.EarlyStopping(min_epochs=50, patience=10, threshold=0.001)

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=5000,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop]
    )

    smodel.fit(Xt, yt)
    ypred = smodel.predict(Xp)

    plots(smodel, yt, yp, ypred, it, ip)
# end


def seq2seq():
    Xt, yt, Xp, yp, it, ip = load_data()

    input_size = Xt.shape[2]  # (batch, seq, data)
    hidden_size = 9
    output_size = yt.shape[2]  # (batch, seq, data)

    print(f" input_size: {input_size}")
    print(f"output_size: {output_size}")

    tmodule = nn.Sequential(
        nnx.Probe("input"),             # (B, L, input_size)
        nnk.LSTM(input=input_size,
                 units=input_size,
                 return_sequence=False,
                 activation='tanh'),
        nnx.Probe("lstm1"),             # (B, input_size)
        # nn.Tanh(),
        nnk.Dense(input=input_size,
                  units=hidden_size,
                  activation='relu'),
        nnx.Probe("dense"),             # (B, hidden_size)
        nnk.RepeatVector(12),
        nnx.Probe("repeat"),            # (B, 12, hidden_size)
        nnk.LSTM(input=hidden_size,
                 units=input_size,
                 return_sequence=True,
                 activation='tanh'),
        nnx.Probe("lstm2"),             # (B, 12, input_size)
        # nn.Tanh(),
        nnk.TimeDistributed(nnk.Dense(input=input_size, units=2*output_size, activation="linear")),
        nnx.Probe("tdist"),             # (B, 12, output_size)
    )

    early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.01)

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=5000,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop]
    )

    smodel.fit(Xt, yt)
    ypred = smodel.predict(Xp)

    plots(smodel, yt, yp, ypred, it, ip)


def main():
    # bilstm()
    seq2seq()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
