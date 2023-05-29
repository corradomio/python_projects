import logging.config
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpyx as npx
import pandas as pd
import pandasx as pdx
import torch.nn as nn
import torchx.nn as nnx
from sklearnx.preprocessing import StandardScaler
from sktime.utils.plotting import plot_series


class TSModel1(nnx.Module):
    def __init__(self, slots, **kwargs):
        self.slots = slots
        super().__init__(
            model=[
                nnx.LSTM(input_size=1, hidden_size=32, num_layers=2, dropout=.2, bidirectional=True),
                nnx.DropDimensions(),
                nn.ReLU(),
                nn.Linear(in_features=2*64*slots, out_features=16*slots),
                nn.ReLU(),
                nn.Linear(in_features=16*slots, out_features=1),
            ],
            **kwargs
        )

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.Xh = None
        self.yh = None

    def fit(self, X: Optional[np.ndarray], y: np.ndarray, batch_size=None, epochs=None):
        Xs = self.x_scaler.fit_transform(X)
        ys = self.y_scaler.fit_transform(y)
        # Xs = X
        # ys = y

        self.Xh = Xs
        self.yh = ys

        lu = npx.UnfoldLoop(self.slots, use_X=False)
        Xt, yt = lu.fit_transform(Xs, ys)

        super().fit(Xt, yt, batch_size, epochs)
        return self

    def predict(self, X: Optional[np.ndarray], fh: int = 0) -> np.ndarray:
        Xs = self.x_scaler.transform(X)
        if fh == 0:
            fh = len(Xs)

        lp = npx.UnfoldPredictor(self.slots, use_X=False)
        ys = lp.fit(self.Xh, self.yh).transform(Xs, fh)

        for i in range(fh):
            Xt = lp.step(i)
            yt = super().predict(Xt)
            ys[i] = yt[0]
        # end

        y = self.y_scaler.inverse_transform(ys)
        # y = ys
        return y
    # end



def main():
    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        datetime=('Month', '%Y-%m', 'M'),
        ignore=['Month'],
        index=['Month'])
    print(len(df))

    y = df[["#Passengers"]]

    # plot_series(y, labels=['#Passengers'])
    # plt.show()

    n = 36

    ya: np.ndarray = y.values.astype('float32')
    y_train = ya[:-n]
    y_test = ya[-n:]

    model = TSModel1(12, batch_size=16, epochs=5000, log_epochs=100)
    model.fit(None, y_train)
    ya_pred = model.predict(None, n)
    ya_pred = ya_pred.reshape(-1)

    y_pred = pd.Series(ya_pred, index=y.index[-n:])

    plot_series(y, y_pred, labels=['y', 'y_pred'])
    plt.show()
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
