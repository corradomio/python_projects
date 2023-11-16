import numpy as np
import scipy.optimize as spo
from pandas import DataFrame

import numpyx as npx
from .base import BaseEncoder


#
# Additive model            Y(t) = T(t) + S(t) + e(t)
# Multiplicative model      Y(t) = T(t) * S(t) * e(t)
#                       ->  log(Y(t) = log(T(t)) + log(S(t)) + log(e(t))

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class Detrend:
    def __init__(self):
        pass

    def fit(self, x, y):
        ...

    def transform(self, x, y):
        ...

    def inverse_transform(self, x, y):
        ...


# ---------------------------------------------------------------------------
# LinearDetrend
# ---------------------------------------------------------------------------

def poly1(x, a0, a1): return a0 + a1*x;


class LinearDetrend(Detrend):

    def __init__(self):
        super().__init__()
        self.params = None

    def fit(self, x, y):
        self.params = spo.curve_fit(poly1, x, y)[0]

    def transform(self, x, y):
        trend = poly1(x, *self.params)
        return y - trend

    def inverse_transform(self, x, y):
        trend = poly1(x, *self.params)
        return y + trend


class DetrendTransform(BaseEncoder):

    def __init__(self, columns=None, method='lin', copy=True):
        super().__init__(columns, copy)
        self.method = method
        self._detrend = {}
        self._start = None

    def fit(self, X: DataFrame):
        X = self._check_X(X)
        self._start = X.index[0]
        xc = np.arange(len(X))

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            if self.method == 'li':
                self._detrend[col] = LinearDetrend()

            self._detrend[col].fit(xc, yc)
        # end
        return self
    # end

    def transform(self, X) -> DataFrame:
        X = self._check_X(X)
        xi = np.array((X.index - self._start))

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            yt = self._detrend[col].transform(xi, yc)

            X[col] = yt
        return X

    def inverse_transform(self, X) -> DataFrame:
        X = self._check_X(X)
        xi = X.index - self._start

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            yi = self._detrend[col].inverse_transform(xi, yc)

            X[col] = yi
        return X
# end


class SeasonalityInfo:

    def __init__(self, fftc):
        n = min(8, len(fftc))
        m = npx.chop(np.abs(fftc))
        s = npx.argsort(m, desc=True)[0:n]
        self.seasonality = s
        self.ampl = npx.chop(np.abs(fftc[s]))
        self.phase = np.angle(fftc[s])
        pass

    def deseasonalize(self, y, offset, order):
        y = y.copy()
        x = np.arange(offset, offset+len(y), dtype=float)
        for i in range(order):
            n = self.seasonality[i]
            ampl = self.ampl[i]/10
            phase = self.phase[i]
            y -= ampl*np.cos(phase + 2*np.pi*x/n)
        return y


class SeasonalityTransform(BaseEncoder):
    def __init__(self, columns=None, order=1, copy=True):
        super().__init__(columns, copy)
        self.order = order
        self._params = {}
        self._start = None

    def fit(self, X: DataFrame):
        X = self._check_X(X)
        self._start = X.index[0]

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            # fftc = scipy.fft.rfft(yc)
            fftc = np.fft.rfft(yc)

            self._params[col] = SeasonalityInfo(fftc)
        # end
        return self

    def transform(self, X: DataFrame):
        X = self._check_X(X)
        start = X.index[0] - self._start

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            seasonality = self._params[col]

            yt = seasonality.deseasonalize(yc, start, self.order)

            X[col] = yt
        # end
        return X
    # end
# end


DeseasonalizingTranform = SeasonalityTransform

