import numpy as np
from numpy import zeros


# ---------------------------------------------------------------------------
# ashuffle
# ---------------------------------------------------------------------------

def ashuffle(*arr_list):
    """
    Shuffle a list of vectors.
    It uses the same order for vectors in sequence with the same length

    :param arr_list: list of arrays
    :return: list of shuffled arrays in the same input order
    """
    n = -1
    idxs = None
    res = []
    for arr in arr_list:
        if len(arr) != n:
            n = len(arr)
            idxs = np.arange(n)
            np.random.shuffle(idxs)
        arr = arr[idxs]
        res.append(arr)
    # end
    return res


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

# def is_pos_def(m: ndarray) -> bool:
#     """Check if the matrix is positive definite"""
#     ev = eigvals(m)
#     return all(ev > 0)


# def is_symmetric(a: ndarray, tol=1e-8) -> bool:
#     """Check if the matrix is symmetric"""
#     return all(abs(a - a.T) < tol)


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------

def as_shape(s) -> tuple:
    """Convert int, list[int] into tuple[int]"""
    if isinstance(s, int):
        return (s,)
    elif isinstance(s, list):
        return tuple(s)
    else:
        return s


def shape_dim(s) -> int:
    """Shepe dimension, as 'rank'"""
    if isinstance(s, int):
        return 1
    else:
        return len(s)


def shape_concat(s1, s2) -> tuple:
    """Concatenate two shapes"""
    s1 = as_shape(s1)
    s2 = as_shape(s2)
    return s1 + s2


def shape_size(s) -> int:
    """Compute the number of elements given the shape"""
    if isinstance(s, int):
        return s

    size = 1
    for sz in s:
        size *= sz
    return size


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------

def m2vec(m):
    """
    Convert the matrix m in 3 vectors useful to be used in
    3D scatter plots

    :param np.ndarray x: x vector
    :param np.ndarray y: y vector
    :param np.ndarray m: xy matrix
    :return tuple: xs, ys, zs
    """

    assert m.ndim == 2

    ny, nx = m.shape
    n = nx * ny
    xs = zeros(shape=n)
    ys = zeros(shape=n)
    zs = zeros(shape=n)

    k = 0
    for i in range(ny):
        for j in range(nx):
            xs[k] = j
            ys[k] = i
            zs[k] = m[i, j]
            k += 1
    return xs, ys, zs


def xym2vec(x, y, m):
    """
    Convert the vectors x, y, and the matrix m , in 3 vectors
    useful to be used in 3D scatter plots

    :param np.ndarray x: x vector
    :param np.ndarray y: y vector
    :param np.ndarray m: xy matrix
    :return tuple: xs, ys, zs
    """
    assert len(x.shape) == 1 and len(y.shape) == 1 and len(m.shape) == 2
    assert (y.shape[0], x.shape[0]) == m.shape

    nx = len(x)
    ny = len(y)
    n = nx * ny

    xs = zeros(shape=n)
    ys = zeros(shape=n)
    zs = zeros(shape=n)

    k = 0
    for i in range(ny):
        for j in range(nx):
            xs[k] = x[j]
            ys[k] = y[i]
            zs[k] = m[i, j]
            k += 1
    return xs, ys, zs


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
