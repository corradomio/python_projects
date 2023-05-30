import numpy as np
import numpyx as npx
import torch
import torch.nn as nn
import torchx.nn as nnx
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# data

f = 2*np.pi/12
t = np.arange(0, 12*20, 1)
a1 = .1
b1 = .2
a2 = .3
b2 = .04
a3 = .0
b3 = f

trend = (a1 + b1*t)
amplitude =  (a2 + b2*t)
period = np.sin(a3 + b3*t)

# y = (a1 + b1*t) + (a2 + b2*t)*np.sin(a3 + b3*t)
y = trend + amplitude*period

# ---------------------------------------------------------------------------
# plot

plt.plot(t, y)
plt.show()


# ---------------------------------------------------------------------------
# Xt, yt

X = t.reshape((-1, 1))
y = y.reshape((-1, 1))

lr = npx.LagReshaper(xlags=[0, 1], ylags=[1])

Xt, yt = lr.fit_transform(X, y)

n = 24

Xtr = Xt[:-n]
ytr = yt[:-n]

Xts = Xt[-n:]
yts = yt[-n:]

lr = LinearRegression()
lr.fit(Xtr, ytr)
ypr = lr.predict(Xts)

y_predict = np.zeros_like(yt)
y_predict[:-n] = ytr
y_predict[-n:] = ypr


# ---------------------------------------------------------------------------
# plot

plt.plot(yt[:, 0])
plt.plot(y_predict[:, 0])
plt.show()

# --

plt.plot(yts[:, 0])
plt.plot(ypr[:, 0])
plt.show()


# ---------------------------------------------------------------------------
# predict train

ypr = lr.predict(Xtr)

plt.plot(ytr[:, 0])
plt.plot(ypr[:, 0])
plt.show()



pass



