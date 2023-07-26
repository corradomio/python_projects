from typing import Union
from random import gauss
import numpy as np
import pandas as pd


def white_noise(v=1.):
    return gauss(0., v)


def ar_process(p: Union[int, list[float]], l: 1000):
    if isinstance(p, int):
        p = [1]*p

    coeff = p
    p = len(p)
    x = [0.]*l

    # start
    for t in range(p):
        x[t] = coeff[t]

    # rest
    for t in range(p, l):
        y = 0
        for i in range(p):
            y += coeff[i]*x[t-i]
        x[t] = y + white_noise()
    return np.array(x)


def ma_process(q: Union[int, list[float]], l: 1000, tc=0):
    if isinstance(q, int):
        q = [1]*q

    coeff = q
    q = len(q)
    e = [white_noise() for i in range(l)]
    y = [0.]*l

    # start
    # for t in range(q):
    #     y[t] = e[t]

    for t in range(q, l):
        y[t] = e[t] + t*tc
        for i in range(q):
            y[t] += coeff[i]*e[t-i-1]

    # return pd.Series(y)
    return np.array(y)