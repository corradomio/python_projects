from random import random, shuffle
from typing import cast, Self, Optional, Union
from stdlib import is_instance
from stdlib.dict import dict
from stdlib.jsonx import load
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt, radians, asin

def sq(x): return x*x

def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # R = 6373.0
    R = 6371.0088
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sq(sin(dlat/2)) + cos(lat1) * cos(lat2) * sq(sin(dlon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    # c = 2 * asin(sqrt(a))
    return R * c


def result_reshape(X: np.ndarray, shape) -> np.ndarray:
    if len(X.shape) == 1:
        return X.reshape((1,) + shape)
    else:
        n = len(X)
        return X.reshape((n,) + shape)


def print_array(a: np.ndarray):
    def _pvec(v):
        n = len(v)
        print("{", end="")
        if n >= 1:
            print(v[0], end="")
        for i in range(1, n):
            print(",", v[i], end="")
        print("}", end="")
    def _pmat(M):
        n,m = M.shape
        print("{", end="")
        _pvec(M[0])
        for i in range(1,m):
            print(",")
            _pvec(M[i])
        print("}")
    if len(a.shape) == 1:
        _pvec(a)
        print()
    else:
        _pmat(a)
# end
