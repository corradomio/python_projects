import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
import torch
import torch.nn as nn
import skorch
import torchx.nn as nnx
from sktimex.transform import RNNTrainTransform, RNNPredictTransform

import pandasx as pdx

# hide warnings
warnings.filterwarnings("ignore")

PLOTS_DIR = "./plots"
DATA_DIR = "../data"

TARGET = 'Number of airline passengers'


#
# LSTM:
#   input   (batch, sequence, input_size)
#   output  (batch, sequence, output_size)
#   hidden  (1,     sequence, output_size)
#
# quindi, per poter passare il contesto alla seconda sequenza, e' necessario
# che gli output delle due sequenze sia lo stesso, MA non c'e' problema con
# l'input.
#

class Seq2SeqV1(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        assert isinstance(input_shape, tuple) and len(input_shape) == 2
        assert isinstance(output_shape, tuple) and len(output_shape) == 2

        self.input_shape = input_shape
        self.output_shape = output_shape
        input_size = input_shape[1]
        output_size = output_shape[1]
        self.seq1 = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            batch_first=True
        )
        self.seq2 = nn.LSTM(
            input_size=output_size,
            hidden_size=output_size,
            batch_first=True
        )

    def forward(self, x):
        t, hs = self.seq1(x)
        zero_shape = (x.shape[0], self.output_shape[0], self.output_shape[1])
        z = torch.zeros(zero_shape)
        y, _ = self.seq2(z, hs)
        return y


class Seq2SeqV2(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        assert isinstance(input_shape, tuple) and len(input_shape) == 2
        assert isinstance(output_shape, tuple) and len(output_shape) == 2

        self.input_shape = input_shape
        self.output_shape = output_shape
        input_size = input_shape[1]
        output_size = output_shape[1]
        self.seq1 = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            batch_first=True
        )
        self.seq2 = nn.LSTM(
            input_size=output_size,
            hidden_size=output_size,
            batch_first=True
        )

    def forward(self, x):
        t, hs = self.seq1(x)
        z = t[:, -1:].repeat((1, self.output_shape[0], 1))
        y, _ = self.seq2(z, hs)
        return y


class Seq2SeqV3(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        assert isinstance(input_shape, tuple) and len(input_shape) == 2
        assert isinstance(output_shape, tuple) and len(output_shape) == 2

        self.input_shape = input_shape
        self.output_shape = output_shape
        input_size = input_shape[1]
        output_size = output_shape[1]
        self.seq1 = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            batch_first=True
        )
        self.lin12 = nnx.Linear(input_shape, output_shape)
        self.seq2 = nn.LSTM(
            input_size=output_size,
            hidden_size=output_size,
            batch_first=True
        )

    def forward(self, x):
        t, hs = self.seq1(x)
        z = self.lin12(t)
        y, _ = self.seq2(z, hs)
        return y


def analyze(df, title):
    df_scaled = pdx.MinMaxScaler(quantile=.05).fit_transform(df)
    X, y = pdx.xy_split(df_scaled, target=TARGET)
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    tt = RNNTrainTransform(slots=[0, 12], tlags=1)
    Xt, yt = tt.fit_transform(X_train, y_train)

    pt = RNNPredictTransform(slots=[0, 12], tlags=1)
    yp = pt.fit(X_train, y_train).transform(X_test, None)

    # tmodule = Seq2SeqV1(Xt.shape[1:3], yt.shape[1:3])
    # tmodule = Seq2SeqV2(Xt.shape[1:3], yt.shape[1:3])
    tmodule = Seq2SeqV3(Xt.shape[1:3], yt.shape[1:3])

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

    n = len(yp)
    i = 0
    while i < n:
        Xp = pt.step(i)
        y_pred = smodel.predict(Xp)
        i = pt.update(i, y_pred)

    y_pred = pd.DataFrame(data=yp, columns=[TARGET], index=y_test.index)

    plot_series(y_train[TARGET], y_test[TARGET], y_pred[TARGET], labels=["train", "test", "pred"],
                title=title)
    plt.show()
    pass


def main():
    df = pdx.read_data(
        f"{DATA_DIR}/airline.csv",
        datetime=['Period', '%Y-%m', 'M'],
        index=['Period'],
        ignore=['Period'],
        binary=[]
    )
    analyze(df, "airline")
# end


if __name__ == "__main__":
    main()
