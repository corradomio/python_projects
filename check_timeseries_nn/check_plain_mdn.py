import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
from sktime.utils.plotting import plot_series

import torchx.nn as nnx
from loaddata import *


def main():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data(12)
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    n_mixtures = 8

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 24, 19)
        nn.Flatten(),
        nnx.Probe("flatten"),
        # (*, 24*19)
        nnx.MixtureDensityNetwork(in_features=window_len*input_size, out_features=predict_len*output_size,
                                  hidden_size=window_len*output_size,
                                  n_mixtures=n_mixtures),
        nnx.Probe("mdn"),
        # (*, 24)
        # nnx.Linear(in_features=(window_len, output_size), out_features=(predict_len, output_size)),
        nnx.Probe("last")
        # (*, 12, 1)
    )

    early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        criterion=nnx.MixtureDensityNetworkLoss,
        criterion__out_features=output_size*predict_len,
        criterion__n_mixtures=n_mixtures,
        lr=0.0001,
        callbacks=[early_stop],
        batch_size=6
    )

    smodel.fit(Xt, yt)

    # -------------------------------------------------------------------------------
    # Forecaster usage

    yf_pred = ft.transform(Xs_test_)
    mdn_predictor = nnx.MixtureDensityNetworkPredictor(
        smodel,
        (predict_len, output_size),
        n_mixtures, n_samples=1)

    n = len(yf_pred)
    i = 0
    while i < n:
        Xpf = ft.step(i)

        ypm = mdn_predictor.predict(Xpf)

        i = ft.update(i, ypm)
        pass

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'],
                title="MixtureDensityNetwork (8)")
    plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
