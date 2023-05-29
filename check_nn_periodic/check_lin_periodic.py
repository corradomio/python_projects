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
periodic = np.sin(b3*t)
y = (a1 + b1*t) + (a2 + b2*t)*periodic
y = y.reshape((-1, 1))

n = len(y)

# --

plt.plot(t, y)
plt.show()

# --

X = np.zeros((n, 2))
X[:, 0] = t
X[:, 1] = periodic

Xt, yt = npx.LagReshaper(xlags=[1, 0], ylags=[1]).fit_transform(X, y)

# --

p = s*3

X_train = Xt[:-p]
y_train = yt[:-p]

X_test = Xt[-p:]
y_test = yt[-p:]


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.plot(y_test)
plt.plot(y_pred)
plt.show()

y_pred_all = np.zeros((len(yt), 1))
y_pred_all[:] = np.nan
y_pred_all[-p:] = y_pred

# --

# plt.plot(yt)
# plt.plot(y_pred)
# plt.show()

# --

Xt, yt = npx.UnfoldLoop(steps=12, use_X=True, use_y=True).fit_transform(X, y)


pass