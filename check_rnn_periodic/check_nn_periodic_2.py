import numpy as np
import numpyx as npx
import skorch
import torch
import torch.nn as nn
import torchx
import torchx.nn as nnx

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from periodic_fn import periodic_fn

# --

s = 12
a1 = .1
b1 = .2
a2 = .3
b2 = .04
c2 = .0
d2 = 1 / s

t = np.arange(0, 20 * s, dtype=np.float32)
y = periodic_fn(t, (a1, b1), (a2, b2, c2, d2))

# --

plt.plot(t, y)
plt.show()

# --

trend = (a1 + b1 * t)
amplitude = (a2 + b2 * t)
periodic = np.sin(c2 + d2 * 2 * np.pi * t)

X = np.zeros((len(t), 2), dtype=np.float32)
X[:, 0] = t
X[:, 1] = periodic

y = y.reshape((-1, 1))

# --

Xt, yt = npx.UnfoldLoop(steps=6, xlags=[1, 0], ylags=[1]).fit_transform(X, y)
input_size = Xt.shape[-1]
ouput_size = yt.shape[-1]

p = 3 * s

Xtr = Xt[:-p]
ytr = yt[:-p]

Xts = Xt[-p:]
yts = yt[-p:]


# --

class Model(nnx.Module):

    def __init__(self):
        super().__init__(
            module=
            nnx.LSTM(input_size=input_size,
                     hidden_size=8,
                     output_size=ouput_size,
                     num_layers=1,
                     bidirectional=False)
            ,
            batch_size=16,
            max_epochs=2000,
            log_epochs=100)


# --

model = skorch.NeuralNetRegressor(
    module=nnx.LSTM(
        input_size=input_size,
        hidden_size=8,
        output_size=ouput_size,
        num_layers=1,
        bidirectional=False),
    batch_size=16,
    callbacks=[skorch.callbacks.EarlyStopping(patience=10)],
    max_epochs=2000,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.Adam,
    lr=0.001
)

model = Model()

model.fit(Xtr, ytr)
yp = model.predict(Xts)

y_true = yts[:, -1, :].reshape(-1)
y_pred = yp[:, -1, :].reshape(-1)

plt.plot(y_true)
plt.plot(y_pred)
plt.show()

y_true = ytr[:, -1, :].reshape(-1)
y_pred = model.predict(Xtr)[:, -1, :].reshape(-1)
plt.plot(y_true)
plt.plot(y_pred)
plt.show()

pass
