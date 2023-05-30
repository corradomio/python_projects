import numpy as np
import numpyx as npx
import torch
import torch.nn as nn
import torchx
import torchx.nn as nnx

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# --

s = 12
a1 = .1
b1 = .2
a2 = .3
b2 = .04
a3 = .0
b3 = 2*np.pi/s

t = np.arange(0, 20*s, 1)
n = len(t)
p = 3*s

periodic = np.sin(a3 + b3*t)
amplitude = (a2 + b2*t)
# amplitude = (1 + 3*np.exp(-.01*t))
trend = (a1 + b1*t)
# trend = (1 + 5*np.exp(-.005*t))
y = trend + amplitude*periodic
# y += (a2 + b2*t)*0.5*np.random.random(y.shape)

X = np.zeros((n, 3))
X[:, 0] = t
X[:, 1] = periodic
X[:, 2] = np.sqrt(1 - np.power(periodic, 2))

# X[:, 0] = 1
# X[:, 1] = 1
# X[:, 2] = 1

y = y.reshape((-1, 1))

# --

plt.plot(t, y)
plt.show()

# --

Xt, yt = npx.UnfoldLoop(steps=6, xlags=[1, 0], ylags=[1]).fit_transform(X, y)
input_size = Xt.shape[-1]
ouput_size = yt.shape[-1]

Xtr = Xt[:-p]
ytr = yt[:-p]

Xts = Xt[-p:]
yts = yt[-p:]


# --

class Model(nnx.Module):

    def __init__(self):
        super().__init__(model=[
            nnx.LSTM(input_size=input_size,
                     hidden_size=8,
                     output_size=ouput_size,
                     num_layers=1,
                     bidirectional=False)
            ],
            batch_size=16,
            epochs=2000,
            log_epochs=100)


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