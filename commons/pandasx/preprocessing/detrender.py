from typing import Union

import numpy as np
import pandas as pd
import scipy.optimize as spo

from stdlib import kwparams
from .base import GroupsEncoder


#
# Additive model            Y(t) = T(t) + S(t) + e(t)
# Multiplicative model      Y(t) = T(t) * S(t) * e(t)
#                       ->  log(Y(t) = log(T(t)) + log(S(t)) + log(e(t))

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class Trend:
    def __init__(self):
        pass

    def fit(self, x, y):
        ...

    def transform(self, x, y):
        ...

    def inverse_transform(self, x, y):
        ...


# ---------------------------------------------------------------------------
# Detrend algorithms
# ---------------------------------------------------------------------------

def period_diff(pi: pd.PeriodIndex, start: pd.Period):
    diff = pi - start
    if len(diff) == 0:
        return np.array([])
    if isinstance(diff[0], int):
        return diff
    diff = np.array([e.n for e in diff], dtype=int)
    return diff


class FunctionTrend(Trend):

    def __init__(self, fun):
        super().__init__()
        self.fun = fun
        self.params = None

    def fit(self, x, y):
        x = x.astype(float)
        y = y.astype(float)
        self.params = spo.curve_fit(self.fun, x, y)[0]
        return self

    def transform(self, x, y):
        x = x.astype(float)
        y = y.astype(float)
        trend = self.fun(x, *self.params)
        return y - trend

    def inverse_transform(self, x, y):
        trend = self.fun(x, *self.params)
        return y + trend
# end


def poly1(x, a0, a1): return a0 + a1*x


def power1(x, a0, a1, a2): return a0 + a1*(x**a2);


# ---------------------------------------------------------------------------

class IndentityTrend(Trend):

    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        return self

    def transform(self, x, y):
        return y

    def inverse_transform(self, x, y):
        return y
# end


# ---------------------------------------------------------------------------
# DetrendTransform
# ---------------------------------------------------------------------------

class DetrendTransform(GroupsEncoder):

    def __init__(self, columns=None, method='lin', groups=None, copy=True, **kwargs):
        super().__init__(columns=columns, groups=groups, copy=copy)
        self.method = method
        self.kwargs = kwargs
        self._detrend = {}
        self._start = {}

    def _get_params(self, g):
        return self._detrend[g], self._start[g]

    def _set_params(self, g, params):
        detrend, start = params
        self._detrend[g] = detrend
        self._start[g] = start

    def _compute_params(self, X):
        start = X.index[0]
        detrend = {}

        X = self._check_X(X)
        xc = np.arange(len(X))

        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            if col not in self._detrend:
                detrend[col] = self.create_trend()

            detrend[col].fit(xc, yc)
        # end
        return detrend, start

    def _apply_transform(self, X, params):
        detrend, start = params
        X = X.copy()
        xi = period_diff(X.index, start)

        for col in self._get_columns(X):
            yc = X[col].to_numpy()
            yt = detrend[col].transform(xi, yc)
            X[col] = yt

        return X

    def _apply_inverse_transform(self, X, params):
        detrend, start = params
        X = X.copy()
        xi = period_diff(X.index, start)

        for col in self._get_columns(X):
            yc = X[col].to_numpy()
            yt = detrend[col].inverse_transform(xi, yc)
            X[col] = yt

        return X

    def create_trend(self):
        params = kwparams(self.kwargs, 'method')
        if self.method is None:
            return IndentityTrend()
        elif self.method == 'lin':
            return FunctionTrend(poly1)
        elif self.method == 'power':
            return FunctionTrend(power1)
        elif self.method == 'indentity':
            return IndentityTrend()
        else:
            raise ValueError(f"Unsupported detrend method {self.method}")
    # end
# end


# ---------------------------------------------------------------------------
# StepwiseMinMaxInfo
# ---------------------------------------------------------------------------

def select_bound(xi, x, y):
    n = len(x)
    if xi < x[0]:
        sp = x[1] - x[0]
        dx = int(((x[0] - xi) + (sp - 1))/sp)
        xp = x[0] - sp*dx
        return y[0] + (y[1] - y[0]) * (xp - x[0]) / (x[1] - x[0])

    for i in range(n-1):
        if x[i] <= xi <= x[i+1]:
            return y[i]

    if xi > x[-1]:
        sp = x[-1] - x[-2]
        dx = int(((xi - x[-1]) + (sp - 1))/sp)
        xp = x[-1] + sp*dx
        return y[-2] + (y[-1] - y[-2]) * (xp - x[-2]) / (x[-1] - x[-2])

class StepwiseMinMaxInfo:
    def __init__(self):
        self.ampl = 1.
        self.x = None
        self.x_range = None
        self.lower_bound = None
        self.upper_bound = None
    # end

    def fit(self, x, lb, ub):
        self.x = x
        self.x_range = min(x), max(x)
        self.y_range = min(lb), max(ub)
        self.lower_bound = lb
        self.upper_bound = ub
        return self

    def transform(self, x, y):
        x = x.astype(float)
        y = y.astype(float)

        a = self.ampl
        lb = self._lower_bound(x)
        ub = self._upper_bound(x)

        yt = a*(y - lb)/(ub - lb)
        return yt

    def _lower_bound(self, x):
        n = len(x)
        lb = np.zeros_like(x, dtype=float)
        for i in range(n):
            lb[i] = select_bound(x[i], self.x, self.lower_bound)
        return lb

    def _upper_bound(self, x):
        n = len(x)
        lb = np.zeros_like(x, dtype=float)
        for i in range(n):
            lb[i] = select_bound(x[i], self.x, self.upper_bound)
        return lb

    def inverse_transform(self, x, y):
        a = self.ampl
        x = x.astype(float)
        y = y.astype(float)

        lb = self._lower_bound(x)
        ub = self._upper_bound(x)

        yt = y/a*(ub - lb) + lb
        return yt
# end


# ---------------------------------------------------------------------------
# PiecewiseMinMaxInfo
# ---------------------------------------------------------------------------

def interpolate(xi, x, y):
    n = len(x)
    if xi < x[0]:
        return y[0] + (y[1]-y[0])*(xi-x[0])/(x[1]-x[0])

    for i in range(n-1):
        if x[i] <= xi <= x[i+1]:
            return y[i] + (y[i+1]-y[i])*(xi-x[i])/(x[i+1]-x[i])

    if xi > x[-1]:
        return y[-2] + (y[-1] - y[-2]) * (xi - x[-2]) / (x[-1] - x[-2])


class PiecewiseMinMaxInfo:
    def __init__(self):
        self.ampl = 1.
        self.x = None
        self.x_range = None
        self.lower_bound = None
        self.upper_bound = None
    # end

    def fit(self, x, lb, ub):
        self.x = x
        self.x_range = min(x), max(x)
        self.y_range = min(lb), max(ub)
        self.lower_bound = lb
        self.upper_bound = ub
        return self

    def transform(self, x, y):
        x = x.astype(float)
        y = y.astype(float)

        a = self.ampl
        lb = self._lower_bound(x)
        ub = self._upper_bound(x)

        yt = a*(y - lb)/(ub - lb)
        return yt

    def _lower_bound(self, x):
        n = len(x)
        lb = np.zeros_like(x, dtype=float)
        for i in range(n):
            lb[i] = interpolate(x[i], self.x, self.lower_bound)
        return lb

    def _upper_bound(self, x):
        n = len(x)
        lb = np.zeros_like(x, dtype=float)
        for i in range(n):
            lb[i] = interpolate(x[i], self.x, self.upper_bound)
        return lb

    def inverse_transform(self, x, y):
        a = self.ampl
        x = x.astype(float)
        y = y.astype(float)

        lb = self._lower_bound(x)
        ub = self._upper_bound(x)

        yt = y/a*(ub - lb) + lb
        return yt
# end


# ---------------------------------------------------------------------------
# MinMaxInfo
# ---------------------------------------------------------------------------

def compute_bound(x, y, upper=False):
    n = len(x)
    x0 = x[0]
    y0 = y[0]
    xn = x[-1]
    yn = y[-1]

    i, j = 0, n-1
    while i < j:

        # check first
        xi = x[i]
        yi = y[i]

        yt = y0 + (xi-x0)/(xn-x0)*(yn-y0)
        if upper and yi > yt or not upper and yi < yt:
            x0 = xi
            y0 = yi
            i, j = 0, n-1
            continue
        # end

        # check last
        xj = x[j]
        yj = y[j]
        yt = y0 + (xj - x0) / (xn - x0)*(yn - y0)
        if upper and yi > yt or not upper and yi < yt:
            xn = xj
            yn = yj
            i, j = 0, n - 1
            continue
        # end

        i += 1
        j -= 1
    # end
    # yt = y0 + (x-x0)/(xn-x0)*(yn-y0)
    return y0 - x0*(yn-y0)/(xn-x0), (yn-y0)/(xn-x0)


class MinMaxInfo:
    def __init__(self):
        self.ampl = 1.
        self.lower_bound = None
        self.upper_bound = None
    # end

    def fit(self, x, lb, ub):
        self.lower_bound = compute_bound(x, lb, upper=False)
        self.upper_bound = compute_bound(x, ub, upper=True)
        return self

    def transform(self, x, y):
        x = x.astype(float)
        y = y.astype(float)

        a = self.ampl
        lb = poly1(x, *self.lower_bound)
        ub = poly1(x, *self.upper_bound)

        yt = a*(y - lb)/(ub - lb)
        return yt

    def inverse_transform(self, x, y):
        a = self.ampl
        x = x.astype(float)
        y = y.astype(float)

        lb = poly1(x, *self.lower_bound)
        ub = poly1(x, *self.upper_bound)

        yt = y/a*(ub - lb) + lb
        return yt
# end


# ---------------------------------------------------------------------------
# LinearMinMaxScaler
# ---------------------------------------------------------------------------

class LinearMinMaxScaler(GroupsEncoder):
    def __init__(self, columns=None, feature_range=(0, 1), *,
                 sp=12, method=False,
                 groups=None, copy=True):
        super().__init__(columns=columns, groups=groups, copy=copy)
        self.feature_range = feature_range
        self.sp = sp    # seasonality period
        self.method = method
        self._minmax = {}
        self._start = {}

    def _get_params(self, g):
        return self._minmax[g], self._start[g]

    def _set_params(self, g, params):
        mmi, start = params
        self._minmax[g] = mmi
        self._start[g] = start

    def _compute_params(self, X):
        sp = self.sp
        n = len(X)
        minmax = {}
        start = X.index[0]

        for col in self._get_columns(X):
            lower_bounds = []
            upper_bounds = []
            positions = []

            yc = X[col].to_numpy()

            for i in range(0, n, sp):
                lower_bounds.append(yc[i:i + sp].min())
                upper_bounds.append(yc[i:i + sp].max())
                positions.append(i)
            # end

            mmi = self._create_mmi()
            mmi.fit(positions, lower_bounds, upper_bounds)

            minmax[col] = mmi
        # end
        return minmax, start
    # end

    def _create_mmi(self):
        if self.method is None:
            return MinMaxInfo()
        elif self.method == 'piecewise':
            return PiecewiseMinMaxInfo()
        elif self.method == 'stepwise':
            return StepwiseMinMaxInfo()
        else:
            raise ValueError(f"Unsupported method {self.method}")

    def _apply_transform(self, X, params):
        minmax, start = params
        X = X.copy()
        xi = period_diff(X.index, start)

        ymin, ymax = self.feature_range

        for col in self._get_columns(X):
            if col not in minmax:
                continue

            yc = X[col].to_numpy()
            yt = minmax[col].transform(xi, yc)
            # apply feature_range
            yt = ymin + (ymax-ymin)*yt
            X[col] = yt
        # end
        return X

    def _apply_inverse_transform(self, X, params):
        minmax, start = params
        X = X.copy()
        xi = period_diff(X.index, start)

        ymin, ymax = self.feature_range

        for col in self._get_columns(X):
            if col not in minmax:
                continue

            yc = X[col].to_numpy()
            # invert feature_range
            yt = (yc - ymin)/(ymax - ymin)
            yt = minmax[col].inverse_transform(xi, yt)
            X[col] = yt
        # end
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
