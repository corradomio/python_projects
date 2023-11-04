import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series

import torchx as tx
import torchx.nn as nnx
import pandas as pd
import pandasx as pdx
import skorch
import skorchx
import sktimex

class Seq2Seq(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_size=0):
        super().__init__()

        assert isinstance(input_shape, tuple) and len(input_shape) == 2
        assert isinstance(output_shape, tuple) and len(output_shape) == 2
        assert isinstance(hidden_size, int)

        hidden_size = hidden_size if hidden_size > 0 else output_shape[1]

        # input_shape  = (i_sequence_len, n_i_features)
        # hidden_size: int
        # output_shape = (o_sequence_len, n_o_features)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size


class Seq2Seq1(Seq2Seq):
    def __init__(self, input_shape, output_shape, hidden_size=0):
        super().__init__(input_shape, output_shape, hidden_size)
        hidden_size = self.hidden_size

        self.rnn1 = nnx.LSTM(input_size=input_shape[1], hidden_size=hidden_size, return_sequence=None, return_state=True)
        self.rnn2 = nnx.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.lin = None if hidden_size == output_shape[1] else nnx.Linear((output_shape[0], hidden_size), output_shape)

        self._zcache = {}

    def forward(self, x: torch.Tensor):
        hs = self.rnn1(x)
        z = self.zeros(x)
        t = self.rnn2(z, hs)
        t = self.lin(t) if self.lin else t
        return t

    def zeros(self, x):
        n = len(x)
        if n not in self._zcache:
            self._zcache[n] = torch.zeros((n, self.output_shape[0], self.hidden_size), dtype=x.dtype).to(x.device)
        return self._zcache[n]


class Seq2Seq2(Seq2Seq):
    def __init__(self, input_shape, output_shape, hidden_size=16):
        super().__init__(input_shape, output_shape, hidden_size)
        hidden_size = self.hidden_size

        self.rnn1 = nnx.LSTM(input_size=input_shape[1], hidden_size=hidden_size, return_sequence=False, return_state=True)
        self.rnn2 = nnx.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.lin = None if hidden_size == output_shape[1] else nnx.Linear((output_shape[0], hidden_size), output_shape)

    def forward(self, x: torch.Tensor):
        t, hs = self.rnn1(x)
        z = tx.time_repeat(t, self.output_shape[0])
        t = self.rnn2(z, hs)
        t = self.lin(t) if self.lin else t
        return t


class Seq2Seq3(Seq2Seq):
    def __init__(self, input_shape, output_shape, hidden_size=0):
        super().__init__(input_shape, output_shape, hidden_size)
        hidden_size = self.hidden_size

        self.rnn1 = nnx.LSTM(input_size=input_shape[1], hidden_size=hidden_size, return_sequence=True, return_state=True)
        self.lin12 = nnx.Linear((input_shape[0], hidden_size), (output_shape[0], hidden_size))
        self.rnn2 = nnx.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.lin = None if hidden_size == output_shape[1] else nnx.Linear((output_shape[0], hidden_size), output_shape)

    def forward(self, x: torch.Tensor):
        t, hs = self.rnn1(x)
        z = self.lin12(t)
        t = self.rnn2(z, hs)
        t = self.lin(t) if self.lin else t
        return t


class Seq2Seq4(Seq2Seq):
    def __init__(self, input_shape, output_shape, hidden_size=0):
        super().__init__(input_shape, output_shape, hidden_size)
        hidden_size = self.hidden_size

        self.rnn = nnx.LSTM(input_size=input_shape[1], hidden_size=hidden_size,
                            return_sequence=True)
        self.lin = nnx.Linear(in_features=(input_shape[0], hidden_size), out_features=output_shape)

    def forward(self, x: torch.Tensor):
        t = self.rnn(x)
        t = self.lin(t)
        return t


def main():
    # TARGET = 'COAST'
    # PAST = 144
    # FUTURE = 24
    # df = pdx.read_data("./data/ercot_data.csv")
    TARGET = 'Number of airline passengers'
    PAST = 48
    FUTURE = 12

    # read data
    df = pdx.read_data("./data/airline.csv", ignore='Period')
    y = df[[TARGET]]

    # scale data
    scaler = pdx.MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    y = df_scaled[[TARGET]]

    # train/test
    y_train, y_test = pdx.train_test_split(y, test_size=2*FUTURE)

    # train transform
    tt = sktimex.RNNTrainTransform(lags=PAST, tlags=FUTURE)
    Xt, yt = tt.fit_transform(y_train)

    # module = Seq2Seq1((144, 1), (24, 1))
    # module = Seq2Seq2((144, 1), (24, 1))
    # module = Seq2Seq3((PAST, 1), (FUTURE, 1))
    module = Seq2Seq4((PAST, 1), (FUTURE, 1))

    early_stop = skorchx.callbacks.EarlyStopping(min_epochs=0, patience=10, threshold=0, monitor="valid_loss")

    net = skorch.NeuralNetRegressor(
        # criterion
        # optimizer
        # iterator_train, iterator_valid, dataset, train_split
        optimizer=torch.optim.Adagrad,
        module=module,
        lr=0.001,
        callbacks=[early_stop],
        callbacks__print_log=skorchx.callbacks.PrintLog(delay=3),
        max_epochs=10000,
        batch_size=32
    )

    net.fit(Xt, yt)

    # predict transform
    pt = sktimex.RNNPredictTransform(lags=PAST, tlags=FUTURE)
    nt = len(y_test)
    y_pred = pt.fit(y_train).transform(fh=nt)

    # predict
    i = 0
    while i < nt:
        X0 = pt.step(i)
        y0 = net.predict(X0)
        i = pt.update(i, y0)

    plot_series(y_test[TARGET], y_pred[TARGET], labels=['test', 'pred'])
    plt.show()

    pass


if __name__ == "__main__":
    main()
