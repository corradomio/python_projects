import logging.config

import matplotlib.pyplot as plt
import skorch
import torch
import torch.nn as nn
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

    # -------------------------------------------------------------------------------

    tmodule = nn.Sequential(
        # (*, 24, 19)
        nnx.Probe("input"),
        nnx.Time2Vec(input_size=(window_len, input_size), output_size=120),
        # (*, 24, 139)
        nnx.Probe("time2vec"),
        nnx.LSTM(input_size=(120 + input_size), hidden_size=48, bidirectional=True, return_sequences=True),
        nnx.Probe("lstm"),
        # (*, 24, 2*48)
        nnx.Linear(in_features=(window_len, 48, 2), out_features=(predict_len, output_size)),
        # (*, 24, 1)
        nnx.Probe("last")
    )

    early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.0001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        lr=0.0008,
        optimizer__betas=(0.9, 0.999),
        optimizer__amsgrad=True,
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

    plot_series(ys_train['EASY'], ys_test_['EASY'], yf_pred['EASY'], labels=['train', 'test', 'pred'])
    plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
