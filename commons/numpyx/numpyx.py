from typing import Union

import numpy as np
from numpy import ndarray, ones, zeros, all, abs
from numpy.linalg import eigvals


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
# end


# ---------------------------------------------------------------------------
# fzeros
# fones
# ---------------------------------------------------------------------------

ShapeType = Union[int, list[int], tuple[int]]


def fzeros(n: ShapeType) -> ndarray: return zeros(n, dtype=float)


def fones(n: ShapeType) -> ndarray: return ones(n,  dtype=float)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def is_pos_def(m: ndarray) -> bool:
    """Check if the matrix is positive definite"""
    ev = eigvals(m)
    return all(ev > 0)


def is_symmetric(a: ndarray, tol=1e-8) -> bool:
    """Check if the matrix is symmetric"""
    return all(abs(a-a.T) < tol)


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------

def as_shape(s: ShapeType) -> tuple[int]:
    """Convert int, list[int] into tuple[int]"""
    if isinstance(s, int):
        return (s,)
    elif isinstance(s, list):
        return tuple(s)
    else:
        return s


def shape_dim(s: ShapeType) -> int:
    """Shepe dimension, as 'rank'"""
    if isinstance(s, int):
        return 1
    else:
        return len(s)


def shape_concat(s1: ShapeType, s2: ShapeType) -> tuple[int]:
    """Concatenate two shapes"""
    s1 = as_shape(s1)
    s2 = as_shape(s2)
    return s1 + s2


def shape_size(s: ShapeType) -> int:
    """Compute the number of elements given the shape"""
    if isinstance(s, int):
        return s

    size = 1
    for sz in s:
        size *= sz
    return size


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------

# def cross_product(x: ndarray, y: ndarray) -> ndarray:
#     """
#     Compute the cross product of the two vectors
#     Note: the cross product is a MATRIX where the dot product is a scalar
#
#     :param x: the column vector
#     :param y: the row vector
#     :return: a matrix
#     """
#     n = len(x)
#     m = len(y)
#     p = zeros((n, m))
#
#     for i in range(n):
#         for j in range(m):
#             p[i, j] = x[i]*y[j]
#     return p


# def compose_eigenvectors(v: ndarray, w: ndarray, inverse=False) -> ndarray:
#     """
#     Compute
#
#             v_i*cross(w_i, w_i)
#
#     note that 'cross(w_i, w_i)' is a MATRIX, not a vector
#
#     :param v: eugenvalues
#     :param w: eigenvectors as columns of the matrix
#     :param inverse: if to compute the inverse of the matrix
#     :return:
#     """
#     n = len(v)
#     p = zeros((n, n))
#
#     for i in range(n):
#         li = 1 / v[i] if inverse else v[i]
#         ui = w[:, i]
#         p += li*cross_product(ui, ui)
#     return p


# def correlation_matrix(X: ndarray, m: ndarray=None) -> ndarray:
#     """
#     Compute the correlation matrix between the columns in the matrix X
#
#         cij = E[Xi - E[Xi]]*E[Xj - E[Xj]]
#
#         E[X] = 1/n * SUM ( x_i : i=1..n )
#
#     :param X: matrix (n rows x d columns)
#     :param m: mean, if already computed
#     :return: correlation matrix
#     """
#     assert isinstance(X, ndarray)
#     assert len(X.shape) == 2
#
#     if m is None:
#         m = X.mean(0)
#
#     n, d = X.shape
#
#     y = X - m
#     cm = zeros((d, d))
#     for i in range(d):
#         yi = y[:, i]
#         for j in range(d):
#             yj = y[:, j]
#             cij = dot(yi, yj)
#             cm[i, j] += cij
#     cm /= n
#     return cm


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

# def mean(X: ndarray) -> ndarray:
#     """
#     Compute the column mean of the matrix X
#
#     :param X: matrix (n rows x d columns)
#     :return: column mean
#     """
#     assert isinstance(X, ndarray)
#     assert len(X.shape) == 2
#     return X.mean(0)


# def gaussian(X: ndarray) -> ndarray:
#     """
#     'x' is the dataset, a matrix 'n x m', where n is the number of
#     records, and m the number of features
#
#     :param X:
#     :return:
#     """
#
#     m = mean(X)
#     S = correlation_matrix(X, m)
#     v, w = eig(S)
#     U = compose_eigenvectors(v, w, inverse=True)
#
#     y = dot(X-m, w)
#     return y


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

    assert len(m.shape) == 2

    ny, nx = m.shape
    n = nx*ny
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
    n = nx*ny

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
# filter_outliers
# ---------------------------------------------------------------------------

# def filter_outliers(array: np.ndarray, outlier_std: float) -> np.ndarray:
#     """
#     Replace all values in the dataset that exceed 'outlier_std' with
#     the median
#
#     :param array: array to analyze
#     :param outlier_std: n of Standard Deviation to use to decide if a value is
#         an outlier
#     :return: a copy of the array with the outliers removed
#     """
#     # if outlier_std is 0 , no test is executed
#     if outlier_std == 0:
#         return array
#
#     # because data is modified, it is better to create a copy
#     array = array.copy()
#     mean = np.mean(array, axis=0)
#     std = np.std(array, axis=0)
#     max_value = mean + (outlier_std * std)
#     min_value = mean - (outlier_std * std)
#     median = np.median(array, axis=0)
#
#     array[(array <= min_value) | (array >= max_value)] = median
#     return array
# # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
