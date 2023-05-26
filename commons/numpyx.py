from types import FunctionType
from typing import Optional, Union
import numpy as np
from numpy import ndarray, zeros, dot, mean, asarray, all, abs
from numpy.linalg import eig, eigvals
import csvx


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def load_data(fname: str, ycol=-1, dtype=None, skiprows=0, na: Optional[str]=None):

    if fname.endswith(".arff"):
        data, _, dtype = csvx.load_arff(fname, na=na)
    else:
        data, _ = csvx.load_csv(fname, dtype=dtype, skiprows=skiprows, na=na)

    data = asarray(data)
    nr, nc = data.shape

    if ycol == 0:
        X = data[:, 1:]
        y = data[:, 0]
    elif ycol == -1 or ycol == (nc - 1):
        X = data[:, 0:-1]
        y = data[:, -1]
    else:
        X = data[:, list(range(0, ycol)) + list(range(ycol + 1, nc))]
        y = data[:, ycol]

    if ycol == 0:
        dtypes = list(set(dtype[1:]))
    elif ycol == -1:
        dtypes = list(set(dtype[0:-1]))
    else:
        dtypes = list(set(dtype[0:ycol] + dtype[ycol+1:]))
    if len(dtypes) == 1 and dtypes[0] in ["enum", "ienum", enumerate]:
        X = X.astype(int)

    if dtype[ycol] in ["enum", "ienum", str, int, enumerate]:
        y = y.astype(int)

    return X, y
# end


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------
# (X, y, xslot, yslots) -> Xt, yt
#
# back_step
#   y[-1]             -> y[0]
#   y[-1],X[-1]       -> y[0]
#   y[-1],X[-1],X[0]  -> y[0]
#

def reshape(X: np.ndarray, y: np.ndarray, 
            xslots: list[int] = [0], yslots: list[int] = [], 
            tslots: list[int] = [0]) -> tuple[np.ndarray, np.ndarray]:
    """
    
    :param X:
    :param y:
    :param xslots:
    :param yslots:
    :return:
    """
    assert isinstance(X, (type(None), np.ndarray))
    assert isinstance(y, np.ndarray)
    assert isinstance(xslots, list)
    assert isinstance(yslots, list)
    assert isinstance(tslots, list)
    def _max(l): return 0 if len(l) == 0 else max(l)

    if len(y.shape) == 1:
        y = y.reshape((-1, 1))

    if X is None:
        X = np.zeros((len(y), 0), dtype=y.dtype)
        xslots = []

    assert len(X) == len(y)

    s = max(_max(xslots), _max(yslots))
    t = max(tslots)
    
    mx = X.shape[1]
    my = y.shape[1]
    n = y.shape[0] - s - t
    
    mt = len(xslots)*mx + len(yslots)*my
    mu = len(tslots)*my
    
    Xt = np.zeros((n, mt), dtype=X.dtype)
    yt = np.zeros((n, mu), dtype=y.dtype)
    
    for i in range(n):
        c = 0
        for j in yslots:
            Xt[i, c:c+my] = y[s+i-j]
            c += my
        for j in xslots:
            Xt[i, c:c+mx] = X[s+i-j]
            c += mx
        
        c = 0
        for j in tslots:
            yt[i, c:c+my] = y[s+i+j]
            c += my
    # end
    
    return Xt, yt
# end


# ---------------------------------------------------------------------------
# unroll_loop
# ---------------------------------------------------------------------------
# X[0]      -> (X[0],X[1],X[2],...)
# X[1]      -> (X[1],X[2],X[3],...)
# X[2]      -> (X[2],X[3],X[4],...)
#

def unroll_loop(X: np.ndarray, steps:int = 1) -> np.ndarray:
    """
    Add an extra axes:
    
        (n, m) -> (n-s+1, s, m)
    
    :param X: 
    :param steps: 
    :return: 
    """
    if len(X.shape) == 1:
        X = X.resshape((-1, 1))

    s = steps
    n = X.shape[0] - s + 1
    m = X.shape[1:]

    t = np.zeros((n, s, *m), dtype=X.dtype)

    for i in range(n):
        for j in range(s):
            t[i, j] = X[i+j]

    return t
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
# Mean
# ---------------------------------------------------------------------------

def mean(X: ndarray) -> ndarray:
    """
    Compute the column mean of the matrix X

    :param X: matrix (n rows x d columns)
    :return: column mean
    """
    assert isinstance(X, ndarray)
    assert len(X.shape) == 2
    return X.mean(0)


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------

def cross_product(x: ndarray, y: ndarray) -> ndarray:
    """
    Compute the cross product of the two vectors
    Note: the cross product is a MATRIX where the dot product is a scalar

    :param x: the column vector
    :param y: the row vector
    :return: a matrix
    """
    n = len(x)
    m = len(y)
    p = zeros((n, m))

    for i in range(n):
        for j in range(m):
            p[i, j] = x[i]*y[j]
    return p


def compose_eigenvectors(v: ndarray, w: ndarray, inverse=False) -> ndarray:
    """
    Compute

            v_i*cross(w_i, w_i)

    note that 'cross(w_i, w_i)' is a MATRIX, not a vector

    :param v: eugenvalues
    :param w: eigenvectors as columns of the matrix
    :param inverse: if to compute the inverse of the matrix
    :return:
    """
    n = len(v)
    p = zeros((n, n))

    for i in range(n):
        li = 1 / v[i] if inverse else v[i]
        ui = w[:, i]
        p += li*cross_product(ui, ui)
    return p


def correlation_matrix(X: ndarray, m: ndarray=None) -> ndarray:
    """
    Compute the correlation matrix between the columns in the matrix X

        cij = E[Xi - E[Xi]]*E[Xj - E[Xj]]

        E[X] = 1/n * SUM ( x_i : i=1..n )

    :param X: matrix (n rows x d columns)
    :param m: mean, if already computed
    :return: correlation matrix
    """
    assert isinstance(X, ndarray)
    assert len(X.shape) == 2

    if m is None:
        m = X.mean(0)

    n, d = X.shape

    y = X - m
    cm = zeros((d, d))
    for i in range(d):
        yi = y[:, i]
        for j in range(d):
            yj = y[:, j]
            cij = dot(yi, yj)
            cm[i, j] += cij
    cm /= n
    return cm


def gaussian(X: ndarray) -> ndarray:
    """
    'x' is the dataset, a matrix 'n x m', where n is the number of
    records, and m the number of features

    :param X:
    :return:
    """

    m = mean(X)
    S = correlation_matrix(X, m)
    v, w = eig(S)
    U = compose_eigenvectors(v, w, inverse=True)

    y = dot(X-m, w)
    return y


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

def filter_outliers(array: np.ndarray, outlier_std: float) -> np.ndarray:
    """
    Replace all values in the dataset that exceed 'outlier_std' with
    the median

    :param array: array to analyze
    :param outlier_std: n of Standard Deviation to use to decide if a value is
        an outlier
    :return: a copy of the array with the outliers removed
    """
    # if outlier_std is 0 , no test is executed
    if outlier_std == 0:
        return array

    # because data is modified, it is better to create a copy
    array = array.copy()
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    max_value = mean + (outlier_std * std)
    min_value = mean - (outlier_std * std)
    median = np.median(array, axis=0)

    array[(array <= min_value) | (array >= max_value)] = median
    return array
# end


def weighted_absolute_percentage_error(v_true: np.ndarray, v_pred: np.ndarray):
    if is_vector(v_true):
        v_true = v_true.reshape(len(v_true))
        v_pred = v_pred.reshape(len(v_pred))
        return _wape_score(v_true, v_pred)

    wape = 0
    for i in range(v_true.shape[-1]):
        wape += _wape_score(v_true[:, i], v_pred[:, i])
    return wape
# end


def _wape_score(v_true: np.ndarray, v_pred: np.ndarray):
    total_actual = 0.0
    total_abs_diff = 0.0
    count = 0
    for value in v_true:
        total_actual = total_actual + value
        # prediction = 0.0
        if count == len(v_pred):
            break
        prediction = v_pred[count]
        total_abs_diff = total_abs_diff + np.abs(value - prediction)
        count = count + 1
    if total_actual > 0:
        return total_abs_diff / total_actual
    return total_abs_diff
# end


def multi_r2_score(v_true: np.ndarray, v_pred: np.ndarray):
    if is_vector(v_true):
        v_true = v_true.reshape(len(v_true))
        v_pred = v_pred.reshape(len(v_pred))
        return r2_score(v_true, v_pred)

    r2 = 0
    for i in range(v_true.shape[-1]):
        r2 += r2_score(v_true[:, i], v_pred[:, i])
    return r2
# end


# ---------------------------------------------------------------------------

def is_vector(array: np.ndarray) -> bool:
    """
    Check if  array can be converted in an 1d vector.
    This is possible it the array has a unique dimension or there is a unique
    dimension greater that 1

    :param array: array to analyze
    :retur: True if there is a single dim greater than 1
    """
    return len(strip_shape(array)) == 1
# end


def strip_shape(array: np.ndarray) -> tuple:
    """
    Remove all dimensions equals to 1

    :param array: array to analyze
    :return: the new shape without dimensions equals to 1
    """
    dims = []
    for dim in array.shape:
        if dim > 1:
            dims.append(dim)
    if len(dims) == 0:
        dims = [1]
    return tuple(dims)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
