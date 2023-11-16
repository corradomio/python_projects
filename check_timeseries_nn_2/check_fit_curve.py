import numpy as np
import scipy.optimize as spo

x = np.arange(100)
y = 3 + .3*x


def f(x, a0, a1):
    return a0 + a1*x

params = spo.curve_fit(f, x, y)
print(params[0])


y = 1 + 3*np.exp(-.2*x)
# print(y)


def f(x, a0, a1, a2):
    return a0 + a1*np.exp(a2*x)

params = spo.curve_fit(f, x, y)
print(params[0])

