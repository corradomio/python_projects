from types import FunctionType
from typing import Optional, Union

import numpy
import numpy as np
from numpy import ndarray, zeros, dot, mean, asarray, all, abs
from numpy.linalg import eig, eigvals


# ---------------------------------------------------------------------------
# LinearTrainTransform
# LinearPredictTransform
# ---------------------------------------------------------------------------
# (X, y, xslot, yslots) -> Xt, yt
#
# back_step
#   y[-1]             -> y[0]
#   y[-1],X[-1]       -> y[0]
#   y[-1],X[-1],X[0]  -> y[0]
#

def _max(l):
    return 0 if len(l) == 0 else max(l)


class LinearTrainTransform:
    def __init__(self, xlags: list[int] = [0], ylags: list[int] = [], tlags=[0]):
        assert isinstance(xlags, list)
        assert isinstance(ylags, list)
        assert isinstance(tlags, list)
        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags

    def __len__(self):
        return max(_max(self.xlags), _max(self.ylags))

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)

        xlags = self.xlags
        ylags = self.ylags
        tlags = self.tlags

        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)
            xlags = []

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        assert len(X) == len(y)

        s = max(_max(xlags), _max(ylags))
        t = max(tlags)
        r = s + t

        mx = X.shape[1]
        my = y.shape[1]
        n = y.shape[0] - r

        mt = len(xlags) * mx + len(ylags) * my
        mu = len(tlags) * my

        Xt = np.zeros((n, mt), dtype=X.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in ylags:
                Xt[i, c:c + my] = y[s + i - j]
                c += my
            for j in xlags:
                Xt[i, c:c + mx] = X[s + i - j]
                c += mx

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[s + i + j]
                c += my
        # end

        return Xt, yt
    # end

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
# end


class LinearPredictTransform:
    def __init__(self, xlags: list[int] = [0], ylags: list[int] = [], tlags=[0]):
        assert isinstance(xlags, list)
        assert isinstance(ylags, list)
        assert isinstance(tlags, list)
        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags
        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None
    # end

    def __len__(self):
        return max(_max(self.xlags), _max(self.ylags))

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)

        self.Xh = X
        self.yh = y
        return self

    def transform(self, X: np.ndarray, fh: int) -> np.ndarray:
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(fh, int)

        Xh = self.Xh
        yh = self.yh
        xlags = self.xlags
        ylags = self.ylags
        tlags = self.tlags

        if X is None:
            X = np.zeros((len(yh), 0), dtype=Xh.dtype)

        s = max(_max(xlags), _max(ylags))
        t = max(tlags)

        mx = Xh.shape[1]
        my = yh.shape[1]
        n = yh.shape[0] - s - t

        mt = len(xlags) * mx + len(ylags) * my
        mu = len(tlags) * my

        yp = np.zeros((fh, mu), dtype=yh.dtype)
        Xt = np.zeros((1, mt), dtype=Xh.dtype)

        self.Xp = X
        self.yp = yp
        self.Xt = Xt

        return yp
    # end

    def _xat(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]
    def _yat(self, i):
        return self.yh[i,0] if i < 0 else self.yp[i,0]

    def step(self, i) -> np.ndarray:
        xat = self._xat
        yat = self._yat
        xlags = self.xlags
        ylags = self.ylags
        mx = self.Xh.shape[1]
        my = self.yh.shape[1]
        Xt = self.Xt

        c = 0
        for j in ylags:
            Xt[0, c:c + my] = yat(i - j)
            c += my
        for j in xlags:
            Xt[0, c:c + mx] = xat(i - j)
            c += mx

        return Xt
    # end

    def fit_transform(self, X, y):
        raise NotImplemented()
# end


LagTrainTransform = LinearTrainTransform
LagPredictTransform = LinearPredictTransform


# ---------------------------------------------------------------------------
# RNNTrainTransform
# RNNPredictTransform
# ---------------------------------------------------------------------------
# X[0]      -> (X[0],X[1],X[2],...)
# X[1]      -> (X[1],X[2],X[3],...)
# X[2]      -> (X[2],X[3],X[4],...)
#
#   y[-1]            -> y[0]
#   X[-1]            -> y[0]
#   X[-1],y[-1]      -> y[0]
#   X[-1],y[-1],X[0] -> y[0]
#
# xlags: [], [1], [0], [0,1]
# ylags: [], [1]

class RNNTrainTransform:
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = max(_max(xlags), _max(ylags))

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)
        return self

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)
        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)
        if y is None:
            y = np.zeros((len(X), 0), dtype=y.dtype)

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = t + (s - 1)

        n = X.shape[0] - r
        mx = X.shape[1]
        my = y.shape[1]

        mt = mx*len(xlags) + my*len(ylags)
        Xt = np.zeros((n, s, mt), dtype=X.dtype)
        yt = np.zeros((n, s, my), dtype=y.dtype)

        for i in range(n):
            for j in range(s):
                c = 0
                for k in ylags:
                    Xt[i, j, c:c + my] = y[i + j + t - k]
                    c += my
                for k in xlags:
                    Xt[i, j, c:c + mx] = X[i + j + t - k]
                    c += mx

                yt[i, j] = y[i + j + t]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X, y)
# end


class RNNPredictTransform:
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = max(_max(xlags), _max(ylags))

        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None
    # end

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)
        if y is None:
            y = np.zeros((len(X), 0), dtype=y.dtype)

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        assert len(X) == len(y)
        return self
    # end

    def transform(self, X: np.ndarray, fh: int = 0):
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        xlags = self.xlags
        ylags = self.ylags
        y = self.yh

        if fh == 0: fh = len(X)
        if X is None:
            X = np.zeros((fh, 0), dtype=y.dtype)

        self.Xp = X

        s = self.steps
        t = self.t
        mx = X.shape[1]
        my = y.shape[1]

        mt = mx*len(xlags) + my*len(ylags)
        Xt = np.zeros((1, s, mt), dtype=X.dtype)
        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xt = Xt
        self.yp = yp
        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i]

    def step(self, i):
        atx = self._atx
        aty = self._aty

        X = self.Xh
        y = self.yh
        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        mx = X.shape[1]
        my = y.shape[1]

        Xt = self.Xt

        for j in range(s):
            c = 0
            for k in ylags:
                Xt[0, j, c:c + my] = aty(i + j - k - s + 1)
                c += my
            for k in xlags:
                Xt[0, j, c:c + mx] = atx(i + j - k - s + 1)
                c += mx
        # end

        return Xt
    # end
# end


UnfoldLoop = RNNTrainTransform
UnfoldPreparer = RNNPredictTransform


# ---------------------------------------------------------------------------
# CNNTrainTransform
# CNNPredictTransform
# ---------------------------------------------------------------------------
# N, Channels, Length

class CNNTrainTransform:
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = max(_max(xlags), _max(ylags))

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)
        return self
    # end

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)
        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)
        if y is None:
            y = np.zeros((len(X), 0), dtype=y.dtype)

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = max(self.t, self.steps)

        n = X.shape[0] - r
        mx = X.shape[1]
        my = y.shape[1]

        mt = mx*len(xlags) + my*len(ylags)
        Xt = np.zeros((n, mt, s), dtype=X.dtype)
        yt = np.zeros((n, my), dtype=y.dtype)

        for i in range(n):
            c = 0
            for k in ylags:
                for j in range(s):
                    Xt[i, c:c + my, j] = y[i + j + t - k]
                c += my
            for k in xlags:
                for j in range(s):
                    Xt[i, c:c + mx, j] = X[i + j + t - k]
                c += mx

            yt[i] = y[i + t]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X, y)
# end


class CNNPredictTransform:
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = max(_max(xlags), _max(ylags))

        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (type(None), np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            X = np.zeros((len(y), 0), dtype=y.dtype)
        if y is None:
            y = np.zeros((len(X), 0), dtype=y.dtype)

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        assert len(X) == len(y)
        return self


    def transform(self, X: np.ndarray, fh: int = 0):
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        xlags = self.xlags
        ylags = self.ylags
        y = self.yh

        if fh == 0: fh = len(X)
        if X is None:
            X = np.zeros((fh, 0), dtype=y.dtype)

        self.Xp = X

        s = self.steps
        t = self.t
        mx = X.shape[1]
        my = y.shape[1]

        mt = mx*len(xlags) + my*len(ylags)
        Xt = np.zeros((1, mt, s), dtype=X.dtype)
        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xt = Xt
        self.yp = yp
        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i]

    def step(self, i):
        atx = self._atx
        aty = self._aty

        X = self.Xh
        y = self.yh
        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        mx = X.shape[1]
        my = y.shape[1]

        Xt = self.Xt

        c = 0
        for k in ylags:
            for j in range(s):
                Xt[0, c:c + my, j] = aty(i + j - k)
            c += my
        for k in xlags:
            for j in range(s):
                Xt[0, c:c + mx, j] = atx(i + j - k)
            c += mx

        return Xt
    # end
# end

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
