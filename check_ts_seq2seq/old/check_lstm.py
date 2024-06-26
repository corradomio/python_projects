#
# A LSTM-based seq2seq model for time series forecasting
# https://medium.com/@shouke.wei/a-lstm-based-seq2seq-model-for-time-series-forecasting-3730822301c5
#
import torch
import torch.nn as nn
import skorch
import skorchx
from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series

import torchx.nn as nnx
import torchx as tx
import pandas as pd
import pandasx as pdx
from sktimex.transform import RNNTrainTransform, RNNPredictTransform


class Seq2Seq1(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_size=128):
        # input_shape  = (input_sequence_len,  input_data_len)
        # output_shape = (output_sequence_len, output_data_len)
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size

        # (*, 10, 1) -> (*, 10, 128)
        self.lstm1 = nnx.LSTM(input_size=input_shape[1],
                              hidden_size=hidden_size,
                              return_state=True,
                              return_sequence=True)

        self.lstm2 = nnx.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              return_sequence=True,
                              return_state=False)

    def forward(self, x):
        t, h1 = self.lstm1(x)
        z = torch.zeros((x.shape[0],) + self.output_shape)
        t = self.lstm2(z, h1)
        return t


def main():
    tx.print_shape(None)

    data = pdx.read_data(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv',
        datetime=('Date', '%d-%b-%y', 'D'),
        ignore=['Date'],
        index=['Date'],
        sort=True,
        reindex=False
    )

    prices = data[['Close']]
    scaler = pdx.MinMaxScaler(quantile=0.05)
    prices_scaled = scaler.fit_transform(prices)

    train_data, test_data = pdx.train_test_split(prices_scaled, train_size=0.8)

    lags = 10
    tlags = 2

    tt = RNNTrainTransform(lags=lags, tlags=tlags)
    Xt, yt = tt.fit_transform(y=train_data)

    tmodule = Seq2Seq1((lags, 1), (tlags, 1))

    # ----

    early_stop = skorchx.callbacks.EarlyStopping(min_epochs=0, patience=10, threshold=0, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=100,
        optimizer=torch.optim.Adam,
        lr=0.01,
        callbacks=[early_stop],
        callbacks__print_log=skorchx.callbacks.PrintLog(delay=3),
        batch_size=32
    )
    smodel.fit(Xt, yt)

    pt = RNNPredictTransform(lags=lags, tlags=tlags)
    yp = pt.fit(train_data).transform(len(test_data))
    y_pred = pd.DataFrame(data=yp, columns=['Close'], index=test_data.index)

    n = len(yp)
    i = 0
    while i < n:
        X0 = pt.step(i)
        y0 = smodel.predict(X0)
        i = pt.update(i, y0)

    plot_series(train_data['Close'], test_data['Close'], y_pred['Close'], labels=["train", "test", "pred"],
                title="Apple share price")
    plt.show()
    pass
# end


if __name__ == "__main__":
    main()
