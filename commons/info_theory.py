import numpy as np
from math import log2, sqrt


def rnd_prob(dn):
    p = np.random.rand(*dn)
    return p / p.sum()


def xlogx(x):
    return -x*log2(x) if x != 0 else 0.


def xylogxy(xy, x, y):
    return -xy*log2(xy/(x*y)) if x != 0 or y != 0 else 0.


def max_entropy(n):
    return n*xlogx(1/n)


def entropy(p: list):
    if isinstance(p, list):
        p = np.array(p)
        
    assert isinstance(p, np.ndarray) and len(p.shape) == 1
    p = p/p.sum()
    return sum(xlogx(pi) for pi in p)


def normalized_entropy(p: list):
    e = entropy(p)
    m = max_entropy(len(p))
    return 1 - e/m


def mutual_information(pxy):
    if isinstance(pxy, list):
        pxy = np.array(pxy)

    assert isinstance(pxy, np.ndarray) and len(pxy.shape) == 2

    pxy = pxy/pxy.sum()
    px = pxy.sum(axis=0)
    py = pxy.sum(axis=1)

    mi = 0.
    m, n = pxy.shape
    for i in range(m):
        for j in range(n):
            mi += xylogxy(pxy[i, j], px[j], py[i])
    return mi
# end


def euclidean_norm(p):
    if isinstance(p, list):
        p = np.array(p)
    return sqrt(np.square(p).sum())
