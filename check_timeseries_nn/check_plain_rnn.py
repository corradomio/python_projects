import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
from sktime.utils.plotting import plot_series

import torchx.nn as nnx
from loaddata import *


def lstm1():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.LSTM(input_size=input_size,
                 hidden_size=input_size,
                 bidirectional=True,
                 return_sequences=False), nn.Tanh(),
        nnx.Probe("lstm"),
        # (*, 2*19) because return_sequences=False that is, it returns just the last element
        nnx.Linear(in_features=(input_size, 2), out_features=(predict_len, output_size)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="LSTM[last]")
    plt.show()

    pass


def lstmn():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    output_size = yt.shape[1]  # 24 (batch, seq, data)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.LSTM(input_size=input_size,
                 hidden_size=input_size,
                 bidirectional=False,
                 return_sequences=True), nn.Tanh(),
        nnx.Probe("lstm"),
        # (*, 24, 19) because return_sequences=False that is, it returns just the last element
        nnx.Linear(in_features=(24, input_size), out_features=(24, output_size)),
        nnx.Probe("linear"),
        # (*, 24, 24)
        nnx.Linear(in_features=(24, output_size), out_features=(24, 1)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="LSTM[sequence]")
    plt.show()

    pass


# --


def gru1():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.GRU(input_size=input_size,
                hidden_size=input_size,
                bidirectional=True,
                return_sequences=False), nn.Tanh(),
        nnx.Probe("gru"),
        # (*, 2*19) because return_sequences=False that is, it returns just the last element
        nnx.Linear(in_features=(input_size, 2), out_features=(predict_len, output_size)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="GRU[last]")
    plt.show()

    pass


def grun():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    output_size = yt.shape[1]  # 24 (batch, seq, data)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.GRU(input_size=input_size,
                hidden_size=input_size,
                bidirectional=False,
                return_sequences=True), nn.Tanh(),
        nnx.Probe("gru"),
        # (*, 24, 19) because return_sequences=False that is, it returns just the last element
        nnx.Linear(in_features=(24, input_size), out_features=(24, output_size)),
        nnx.Probe("linear"),
        # (*, 24, 24)
        nnx.Linear(in_features=(24, output_size), out_features=(24, 1)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="GRU[sequence]")
    plt.show()

    pass


# --


def rnn1():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.RNN(input_size=input_size,
                hidden_size=input_size,
                bidirectional=True,
                return_sequences=False), nn.Tanh(),
        nnx.Probe("rnn"),
        # (*, 2*19) because return_sequences=False that is, it returns just the last element
        nnx.Linear(in_features=(input_size, 2), out_features=(predict_len, output_size)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="RNN[last]")
    plt.show()

    pass


def rnnn():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    output_size = yt.shape[1]  # 24 (batch, seq, data)

    input_size = Xt.shape[2]  # 19 (batch, seq, data)
    window_len = Xt.shape[1]  # 24
    output_size = yt.shape[2]  # 1 (batch, seq, data)
    predict_len = yt.shape[1]  # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.RNN(input_size=input_size,
                hidden_size=input_size,
                bidirectional=False,
                return_sequences=True), nn.Tanh(),
        nnx.Probe("lstm"),
        # (*, 24, 19) because return_sequencesimport logging.config
        #
        # import matplotlib.pyplot as plt
        # import skorch
        # import torch
        # import torch.nn as nn
        # from sktime.utils.plotting import plot_series
        #
        # import torchx.nn as nnx
        # from loaddata import *
        #
        #
        # def main():
        #     Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
        #     ft = at.forecaster()
        #     Xp, yp, ip = at.transform(Xs_test_, ys_test_)
        #
        #     input_size = Xt.shape[2]    # 19 (batch, seq, data)
        #     window_len = Xt.shape[1]    # 24
        #     output_size = yt.shape[2]   # 1 (batch, seq, data)
        #     predict_len = yt.shape[1]   # 12
        #
        #     # -------------------------------------------------------------------------------
        #
        #     tmodule = nn.Sequential(
        #         nnx.Probe("input"),
        #         # (*, 24, 19)
        #         nn.Flatten(),
        #         nnx.Probe("flatten"),
        #         # (*, 24*19)
        #         nnx.MixtureDensityNetwork(in_features=window_len*input_size, out_features=predict_len*output_size,
        #                                   hidden_size=window_len*output_size,
        #                                   n_mixtures=8),
        #         nnx.Probe("mdn"),
        #         # (*, 24)
        #         # nnx.Linear(in_features=(window_len, output_size), out_features=(predict_len, output_size)),
        #         nnx.Probe("last")
        #         # (*, 12, 1)
        #     )
        #
        #     early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.001, monitor="valid_loss")
        #
        #     smodel = skorch.NeuralNetRegressor(
        #         module=tmodule,
        #         max_epochs=1000,
        #         optimizer=torch.optim.Adam,
        #         lr=0.0001,
        #         callbacks=[early_stop],
        #         batch_size=6
        #     )
        #
        #     smodel.fit(Xt, yt)
        #
        #     # -------------------------------------------------------------------------------
        #     # Forecaster usage
        #
        #     yf_pred = ft.transform(Xs_test_)
        #
        #     n = len(yf_pred)
        #     i = 0
        #     while i < n:
        #         Xpf = ft.step(i)
        #
        #         ypm = smodel.predict(Xpf)
        #
        #         i = ft.update(i, ypm)
        #         pass
        #
        #     plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
        #                 title="MixtureDensityNetwork (8)")
        #     plt.show()
        #
        #     pass
        #
        #
        # if __name__ == "__main__":
        #     logging.config.fileConfig('../logging_config.ini')
        #     log = logging.getLogger("root")
        #     log.info("Logging system configured")
        #     main()=False that is, it returns just the last element
        nnx.Linear(in_features=(24, input_size), out_features=(24, output_size)),
        nnx.Probe("linear"),
        # (*, 24, 24)
        nnx.Linear(in_features=(24, output_size), out_features=(24, 1)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

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
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = smodel.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="RNN[sequence]")
    plt.show()

    pass


# --

def main():
    lstm1()
    lstmn()
    rnn1()
    rnnn()
    gru1()
    grun()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
