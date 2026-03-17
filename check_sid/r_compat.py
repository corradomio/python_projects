from typing import Union, cast

import numpy as np


def c(*args, dtype=None) -> np.ndarray:
    if len(args) == 0:
        return None
    if dtype is not None:
        return np.array(args, dtype=dtype)
    elif isinstance(args[0], bool):
        return np.array(args, dtype=bool)
    elif isinstance(args[0], int):
        return np.array(args, dtype=int)
    elif isinstance(args[0], float):
        return np.array(args, dtype=float)
    elif isinstance(args[0], np.ndarray) and len(args) == 1:
        return cast(np.ndarray, args[0]).reshape(-1)
    else:
        return np.array(args)


def rbind(*args) -> np.ndarray:
    if len(args) == 0:
        return None
    else:
        return np.array(args)


def cbind(*args) -> np.ndarray:
    if len(args) == 0:
        return None
    else:
        return np.array(args).T


def t(a: np.ndarray) -> np.ndarray:
    return a.T


def matrix(a, nrows, ncols=1, dtype=None) -> np.ndarray:
    assert isinstance(nrows, int)
    assert isinstance(ncols, int)
    assert isinstance(a, (int, float, np.ndarray))
    if isinstance(a, int):
        dtype = int if dtype is None else dtype
        return np.zeros((nrows, ncols), dtype=dtype) + a
    if isinstance(a, float):
        dtype = float if dtype is None else dtype
        return np.zeros((nrows, ncols), dtype=dtype) + a
    else:
        return a.reshape((nrows, ncols))


def length(a: np.ndarray) -> int:
    assert isinstance(a, (list, np.ndarray))
    if isinstance(a, list): return len(a)
    if len(a.shape) == 0: return 0
    if len(a.shape) == 1: return a.shape[0]
    if len(a.shape) == 2: return a.shape[0]*a.shape[1]
    else: raise ValueError(f"Unsupported array with shape {a.shape}")


def seq(start: int, end: int, by: int=1, dtype=int):
    # start:end == seq(start, end, 1)
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(by, int)
    return np.arange(start, end+1, by, dtype=dtype)


def dim(a: np.ndarray) -> tuple:
    return a.shape


def rep(a, each: int=1, dtype=None) -> Union[list, np.ndarray]:
    assert isinstance(a, (int, float, list, np.ndarray))
    if isinstance(a, int):
        dtype = int if dtype is None else dtype
        return np.zeros(each, dtype=dtype) + a
    if isinstance(a, float):
        dtype = float if dtype is None else dtype
        return np.zeros(each, dtype=dtype) + a
    if isinstance(a, list):
        list_of_list = []
        for i in range(each):
            list_of_list.append(a[:])
        return list_of_list
    assert isinstance(a, np.ndarray)
    a = a.reshape(-1)
    n = a.shape[0]
    nr = n*each
    r = np.zeros(nr, dtype=a.dtype)
    for i in range(0, nr, n):
        r[i:i+n] = a
    return r


def which(p):
    return np.where(p)


def expand_grid(m) -> np.ndarray:
    if isinstance(m, np.ndarray):
        m = m.tolist()

    grid = [[]]
    for prefix in m[::-1]:
        ric = []
        for p in prefix:
            for g in grid:
                ric.append(
                    [p] + g
                )
        grid = ric
    # end
    return np.array(grid)


def as_matrix(m):
    return np.array(m)


def list_(a: np.array):
    assert isinstance(a, np.ndarray)
    return a.tolist()


def duplicated(m: np.ndarray):
    n = m.shape[0]
    d = np.zeros(n, dtype=bool)
    for i in range(1, n):
        for j in range(i):
            if np.all(m[i, :] == m[j, :]):
                d[i] = True
                break
    return d
# end
