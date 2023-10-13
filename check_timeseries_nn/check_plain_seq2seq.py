import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
from sktime.utils.plotting import plot_series

import torchx.nn as nnx
from loaddata import *


def rnn():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    hidden_size = 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 24, 19)
        # nnx.Seq2SeqNetwork(input_size=input_size, output_seq=predict_len, hidden_size=input_size, flavour='rnn'),
        nnx.Seq2SeqNetwork(input_size=input_size, output_seq=predict_len, hidden_size=hidden_size, flavour='rnn'),
        nnx.Probe("seq2seq"),
        # (*, 24, 19)
        # nnx.Linear(in_features=(predict_len, input_size), out_features=(predict_len, output_size)),
        nnx.Linear(in_features=(predict_len, hidden_size), out_features=(predict_len, output_size)),
        nnx.Probe("last"),
        # (*, 24, 1)
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=25, threshold=0.001, monitor="valid_loss")

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

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="Seq2seq[RNN]")
    plt.show()

    pass


def gru():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 24, 19)
        nnx.Seq2SeqNetwork(input_size=input_size, output_seq=predict_len, flavour='gru'),
        nnx.Probe("seq2seq"),
        # (*, 24, 19)
        nnx.Linear(in_features=(predict_len, input_size), out_features=(predict_len, output_size)),
        nnx.Probe("last"),
        # (*, 24, 1)
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=25, threshold=0.001, monitor="valid_loss")

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

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="Seq2seq[GRU]")
    plt.show()

    pass


def lstm():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 24, 19)
        nnx.Seq2SeqNetwork(input_size=input_size, output_seq=predict_len, flavour='lstm'),
        nnx.Probe("seq2seq"),
        # (*, 24, 19)
        nnx.Linear(in_features=(predict_len, input_size), out_features=(predict_len, output_size)),
        nnx.Probe("last"),
        # (*, 24, 1)
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=25, threshold=0.001, monitor="valid_loss")

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

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="Seq2seq[LSTM]")
    plt.show()

    pass


def main():
    rnn()
    gru()
    lstm()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
