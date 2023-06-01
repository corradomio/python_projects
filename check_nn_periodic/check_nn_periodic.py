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

# plt.plot(t, y)
# plt.show()

# --

trend = (a1 + b1 * t)
amplitude = (a2 + b2 * t)
periodic = np.sin(c2 + d2 * 2 * np.pi * t)

X = np.zeros((len(t), 2), dtype=np.float32)
X[:, 0] = t
X[:, 1] = periodic

y = y.reshape((-1, 1))

# --

ul = npx.UnfoldLoop(steps=6, xlags=[1, 0], ylags=[1])
Xt, yt = ul.fit_transform(X, y)
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
            module=nnx.LSTM(
                input_size=input_size,
                hidden_size=8,
                output_size=ouput_size,
                num_layers=1,
                bidirectional=False)
            ,
            batch_size=16,
            max_epochs=2000,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            lr=0.001,
            log_epochs=100,
        )


# --
# module,
# criterion,
# optimizer=torch.optim.SGD,
# lr=0.01,
# max_epochs=10,
# batch_size=128,
# iterator_train=DataLoader,
# iterator_valid=DataLoader,
# dataset=Dataset,
# train_split=ValidSplit(5),
# callbacks=None,
# predict_nonlinearity='auto',
# warm_start=False,
# verbose=1,
# device='cpu',
# compile=False,

model = skorch.NeuralNetRegressor(
    module=nnx.LSTM(
        input_size=input_size,
        hidden_size=8,
        output_size=ouput_size,
        num_layers=1,
        bidirectional=False)
    ,
    batch_size=16,
    callbacks=[skorch.callbacks.EarlyStopping(patience=10, monitor="train_loss")],
    max_epochs=2000,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    train_split=None,
    # predict_nonlinearity=None
)

# model = Model()

Xtr1, ytr1 = npx.ashuffle(Xtr, ytr)

model.fit(Xtr1, ytr1)
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
