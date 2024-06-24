import numpy as np
from .transf import Transformer


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class Scaler(Transformer):
    """
    Base class for scalers defined in this module
    """
    def __init__(self):
        super().__init__()
# end


# ---------------------------------------------------------------------------
# IdentityScaler
# ---------------------------------------------------------------------------

class IdentityScaler(Scaler):
    """
    Scaler that does nothing.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, array: np.ndarray):
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array
# end


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler(Scaler):
    """
    Apply a scaling to the array in such way that all valuer are in
    the range [min, max].

    It is possible to apply the scaling globally or per column

    Note: this scaler is sensible to the outliers. To reduce this
          sensibility, configure 'outlier'
    """

    def __init__(self, min=0., max=1., outlier=0., globally=False):
        """
        Initialize the transformer.

        Default range: [0, 1]

        :param min: min range value
        :param max: max range value
        :param outlier: if not zero, the values outside the range [mean-outlier*std, mean+outlier*std]
            will be clipped
        :param globally: if to apply the transformation globally or by column
        """
        super().__init__()

        self.minr = min
        self.diffr = max - min
        self.outlier = outlier

        self.minv = {}
        self.diffv = {}

        self.globally = globally
        self.rank = 0
        assert self.diffr != 0

    def fit(self, X: np.ndarray):
        assert isinstance(X, np.ndarray)
        assert X.ndim <= 2, "Unsupported arrays with ndim > 2"

        if X.ndim == 1:
            self._fit(0, X)
        elif self.globally:
            self._fit(0, X.reshape(-1))
        else:
            for i in range(X.shape[1]):
                self._fit(i, X[:, i])

        self.rank = X.ndim
        return self

    def _fit(self, index: int, X: np.ndarray):
        if self.outlier > 0:
            mean = X.mean()
            stdv = X.std()
            dmin = mean - self.outlier*stdv
            dmax = mean + self.outlier*stdv
        else:
            dmin = X.min()
            dmax = X.max()

        self.minv[index] = dmin
        self.diffv[index] = dmax - dmin
        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        assert self.rank == X.ndim

        if X.ndim == 1 or X.shape[1] == 1:
            T = self._transform(0, X)
        elif self.globally:
            t_shape = X.shape
            X = X.reshape(-1)
            T = self._transform(0, X.reshape(-1))
            T = T.reshape(t_shape)
        else:
            T = np.zeros_like(X)
            for i in range(X.shape[1]):
                T[:, i] = self._transform(i, X[:, i])

        return T

    def _transform(self, index: int, X: np.ndarray):
        minr = self.minr
        diffr = self.diffr

        minv = self.minv[index]
        diffv = self.diffv[index]
        maxv = minv + diffv

        T = X.copy()
        T[T < minv] = minv
        T[T > maxv] = maxv

        if diffv != 0:
            T = minr + (T - minv)*(diffr/diffv)

        return T

    def inverse_transform(self, X):
        assert isinstance(X, np.ndarray)
        assert self.rank == X.ndim

        if X.ndim == 1 or X.shape[1] == 1:
            T = self._inverse_transform(0, X)
        elif self.globally:
            t_shape = X.shape
            T = self._inverse_transform(0, X.reshape(-1))
            T = T.reshape(t_shape)
        else:
            T = np.zeros_like(X)
            rank = X.ndim
            for i in range(rank):
                T[:, i] = self._inverse_transform(i, T[:, i])

        return T

    def _inverse_transform(self, index, T):
        minr = self.minr
        diffr = self.diffr

        minv = self.minv[index]
        diffv = self.diffv[index]

        T[:] = minv + (T - minr) * (diffv / diffr)
        return T

# end


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler(Scaler):
    """
   Apply a scaling to the array in such way that all valuer have
   a normal distribution with parameters 'mean' and 'std'

   It is possible to apply the scaling globally or per column
   """

    def __init__(self, mean: float = 0., std: float = 1., params=None, clip=None, globally=False):
        """
        Initialize the transformer

        Default distribution parameters: mean=0, std=1

        :param mean: mean
        :param std: standard deviation
        :param params: optional[tuple[float, float]] alternative to 'mean'/'std'
        :param clip: optional[min_value, max_value]
        :param globally: if to apply the transformation globally or by column
        """
        super().__init__()

        if params is not None:
            mean, std = params

        self.meanr = mean
        self.stdr = std
        self.clip = clip

        self.meanv = None
        self.stdv = None

        self.globally = globally

    def fit(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)

        globally = self.globally
        if globally:
            self.meanv = array.mean()
            self.stdv = array.std()
        else:
            self.meanv = array.mean(axis=0)
            self.stdv = array.std(axis=0)
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)

        if self.globally:
            scaled = self.meanr + (array - self.meanv) / self.stdv * self.stdr
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.meanr + (array[:, i] - self.meanv[i]) / self.stdv[i] * self.stdr
        # end
        if self.clip is not None:
            minc, maxc = self.clip
            scaled[scaled < minc] = minc
            scaled[scaled > maxc] = maxc
        return scaled

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)

        if self.globally:
            scaled = self.meanv + (array - self.meanr) / self.stdr * self.stdv
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.meanv[i] + (array[:, i] - self.meanr) / self.stdr * self.stdv[i]
        return scaled
# end


NormalScaler = StandardScaler

