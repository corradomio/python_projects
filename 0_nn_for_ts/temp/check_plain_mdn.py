import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrsched
from sktime.utils.plotting import plot_series
import torchx.nn as nnx
from loaddata import *


def main():
    Xt, yt, it, ys_train, Xs_test_, ys_test_, at = load_data()
    ft = at.forecaster()
    Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    input_size = Xt.shape[2]    # 19 (batch, seq, data)
    window_len = Xt.shape[1]    # 24
    output_size = yt.shape[2]   # 1 (batch, seq, data)
    predict_len = yt.shape[1]   # 12

    hidden_size = 20
    n_mixtures = 8

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        nnx.Probe("input"),
        # (*, 24, 19)
        nnx.TimeDistributed(
            # (24, 19)
            nnx.Probe("td.in"),
            nnx.MixtureDensityNetwork(
                in_features=input_size,
                out_features=output_size,
                n_mixtures=n_mixtures,
            ),
            nnx.Probe("td.out"),
        ),
        nnx.Probe("last")
    )

    # early_stop = skorchx.callbacks.EarlyStopping(min_epochs=100, patience=10, threshold=0.0001)
    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0.001, monitor="valid_loss")
    lr_scheduler = skorch.callbacks.LRScheduler(policy=lrsched.CosineAnnealingLR, T_max=1000)

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        criterion=nnx.MixtureDensityNetworkLoss,
        criterion__out_features=output_size,
        criterion__n_mixtures=n_mixtures,
        lr=0.01,
        callbacks=[early_stop, lr_scheduler],
        batch_size=32
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
                title="TemporalConvolutionalNetwork")
    plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('../logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
