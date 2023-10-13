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
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    hidden_size = 250
    n_mixtures = 8

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnk.LSTM(input=input_size,
                 units=input_size,
                 bidirectional=True,
                 return_sequences=False), nn.Tanh(),
        nnx.Probe("lstm"),
        # (*, 2*19) because return_sequence=False and bidirectional=True
        nnk.Dense(input=(input_size, 2), units=hidden_size), nn.Tanh(),
        # (*, 250)
        nnx.Probe("dense"),
        nnx.MixtureDensityNetwork(in_features=hidden_size,
                                  out_features=output_size * predict_len,
                                  n_mixtures=n_mixtures),
        # the output of mixture is a tensor [..mus:8 .., ..sigmas:8.., ..pi:8..]
        nnx.Probe("last")
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0.001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        criterion=nnx.MixtureDensityNetworkLoss,
        criterion__output_size=output_size*predict_len,
        criterion__n_mixtures=n_mixtures,
        lr=0.0001,
        callbacks=[early_stop],
        batch_size=32
    )

    smodel.fit(Xt, yt)

    # -------------------------------------------------------------------------------
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)
    mdn_predictor = nnx.MixtureDensityNetworkPredictor(
        smodel,
        (predict_len, output_size),
        n_mixtures, n_samples=100)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = mdn_predictor.predict(Xpf)

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
