import logging.config
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpyx as npx
import pandas as pd
import pandasx as pdx
import torchx.nn as nnx
from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series


class TSModel1(nnx.Module):
    def __init__(self, *, steps, input_size, hidden_size, **kwargs):
        self.steps = steps
        super().__init__(
            model=[
                nnx.LSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         output_size=1,
                         num_layers=1,
                         dropout=.0,
                         bidirectional=False),
            ],
            **kwargs
        )

        # self.x_scaler = StandardScaler()
        # self.y_scaler = StandardScaler()
        self.Xh = None
        self.yh = None

    def fit(self, X: Optional[np.ndarray], y: np.ndarray, batch_size=None, epochs=None, val=None):
        # Xs = self.x_scaler.fit_transform(X)
        # ys = self.y_scaler.fit_transform(y)
        Xs = X
        ys = y

        self.Xh = Xs
        self.yh = ys

        ul = npx.RNNTrainTransform(self.steps, xlags=[1])
        Xt, yt = ul.fit_transform(Xs, ys)

        if val is not None:
            Xv, yv = val
            # Xv = self.x_scaler.transform(Xv)
            # yv = self.y_scaler.transform(yv)

            Xv, yv = ul.transform(Xv, yv)
            val = (Xv, yv)
        # end

        super().fit(Xt, yt, batch_size=batch_size, epochs=epochs, val=val)
        return self

    def predict(self, X: Optional[np.ndarray], fh: int = 0) -> np.ndarray:
        # Xs = self.x_scaler.transform(X)
        Xs = X
        if fh == 0:
            fh = len(Xs)

        lp = npx.RNNPredictTransform(self.steps, xlags=[1])
        ys = lp.fit(self.Xh, self.yh).transform(Xs, fh)

        for i in range(fh):
            Xt = lp.step(i)
            yt = super().predict(Xt)
            ys[i] = yt[0, -1]
        # end

        # y = self.y_scaler.inverse_transform(ys)
        y = ys
        return y
    # end


def main():
    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        datetime=('Month', '%Y-%m', 'M'),
        ignore=['Month'],
        index=['Month'])
    # print(len(df))

    y = df[["#Passengers"]]
    X = pd.DataFrame({}, index=y.index)

    n = 24

    y_train = y.iloc[:-23]
    y_test = y.iloc[-23:]

    # plot_series(y, labels=['#Passengers'])
    # plt.show()

    # X = pdx.periodic_encode(X, year_scale=[1960, 0.13253012048192758])
    X = pdx.periodic_encode(X)
    X_train = X.iloc[:-23]
    X_test = X.iloc[-23:]

    Xa_train = X_train.values.astype('float32')
    Xa_test = X_test.values.astype('float32')

    ya_train = y_train.values.astype('float32')
    ya_test = y_test.values.astype('float32')

    x_scaler = MinMaxScaler()
    Xa_train = x_scaler.fit_transform(X_train)
    Xa_test = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    ya_train = y_scaler.fit_transform(ya_train)
    ya_test = y_scaler.transform(ya_test)

    input_size = Xa_train.shape[1] + ya_train.shape[1]

    model = TSModel1(steps=20, input_size=input_size, hidden_size=20, batch_size=16, epochs=500, log_epochs=100)
    model.fit(Xa_train, ya_train, val=(Xa_test, ya_test))
    ya_pred = model.predict(Xa_test)
    ya_pred = y_scaler.inverse_transform(ya_pred)
    ya_pred = ya_pred.reshape(-1)

    n = len(ya_pred)
    y_pred = pd.Series(ya_pred, index=y.index[-n:])

    plot_series(y, y_pred, labels=['y', 'y_pred'])
    plt.show()
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
