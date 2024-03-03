# Used to remove the trend
#
# Other methods (sktime):
#
#   Detrender(forecaster=None, model='additive')
#       PolynomialTrendForecaster(degree=1)
#
import numpy as np

from stdlib import kwparams
from .base import GroupsBaseEncoder
from .minmax import period_diff, interpolate_bounds, select_bounds, fit_function, select_seasonal_values
from .minmax import poly1, poly3, power1


#
# Additive model            Y(t) = T(t) + S(t) + e(t)
# Multiplicative model      Y(t) = T(t) * S(t) * e(t)
#                       ->  log(Y(t) = log(T(t)) + log(S(t)) + log(e(t))

# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class Trend:
    def fit(self, x, y):
        ...

    def transform(self, x, y):
        ...

    def inverse_transform(self, x, y):
        ...
# end


# ---------------------------------------------------------------------------
# FunctionTrend
# ---------------------------------------------------------------------------

class FunctionTrend(Trend):

    def __init__(self, fun):
        super().__init__()
        self.fun = fun
        self._trend = None

    def fit(self, x, y):
        x = x.astype(float)
        y = y.astype(float)
        # self.params = spo.curve_fit(self.fun, x, y)[0]
        self._trend = fit_function(self.fun, x, y)
        return self

    def transform(self, x, y):
        x = x.astype(float)
        y = y.astype(float)
        # trend = self.fun(x, *self.params)
        trend = self._trend(x)
        return y - trend

    def inverse_transform(self, x, y):
        # trend = self.fun(x, *self.params)
        trend = self._trend(x)
        return y + trend


# ---------------------------------------------------------------------------
# IndentityTrend
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


# ---------------------------------------------------------------------------
# PiecewiseTrend
# ---------------------------------------------------------------------------

class PiecewiseTrend(Trend):

    def __init__(self, sp, method='mean'):
        super().__init__()
        self.sp = sp
        self.method = method
        self.xy = None

    def fit(self, x, y):
        self.xy = select_seasonal_values(self.sp, x, y, method=self.method, centered=True)
        return self

    def transform(self, x, y):
        trend = interpolate_bounds(self.xy, x)
        return y - trend

    def inverse_transform(self, x, y):
        trend = interpolate_bounds(self.xy, x)
        return y + trend


# ---------------------------------------------------------------------------
# StepwiseTrend
# ---------------------------------------------------------------------------

class StepwiseTrend(Trend):

    def __init__(self, sp, method='mean'):
        super().__init__()
        self.sp = sp
        self.method = method
        self.xy = None

    def fit(self, x, y):
        self.xy = select_seasonal_values(self.sp, x, y, method=self.method)
        return self

    def transform(self, x, y):
        trend = select_bounds(self.xy, x)
        return y - trend

    def inverse_transform(self, x, y):
        trend = select_bounds(self.xy, x)
        return y + trend


# ---------------------------------------------------------------------------
# GlobalTrend
# ---------------------------------------------------------------------------

class ConstantTrend(Trend):

    def __init__(self, sp, method='mean'):
        super().__init__()
        self.sp = sp
        self.method = method
        self.value = None

    def fit(self, x, y):
        if self.method in [None, 'mean']:
            self.value = y.mean()
        elif self.method == 'median':
            self.value = y.median()
        elif self.method == 'min':
            self.value = y.min()
        elif self.method == 'max':
            self.value = y.max()
        else:
            raise ValueError(f'Unsupported method {self.method}')
        return self

    def transform(self, x, y):
        trend = self.value
        return y - trend

    def inverse_transform(self, x, y):
        trend = self.value
        return y + trend
# end


# ---------------------------------------------------------------------------
# DetrendTransform
# ---------------------------------------------------------------------------
# supported methods
#   None                identity
#   identity            identity
#   linear              global linear trend
#   piecewise           seasonal linear trend       (min,max,mean,median)
#   stepwise            seasonal constant trend     (min,max,mean,median)
#   global              remove global value         (min,max,mean,median)
#   poly1               global linear trend
#   power               global exponential trend
#

class DetrendTransformer(GroupsBaseEncoder):

    def __init__(self, columns=None, *,
                 method='linear', sp=12,
                 groups=None, copy=True, **kwargs):
        super().__init__(columns=columns, groups=groups, copy=copy)
        self.method = method
        self.sp = sp  # seasonality period
        self.kwargs = kwargs
        self._detrend = {}
        self._start = {}

    def _get_params(self, g):
        return self._detrend[g], self._start[g]

    def _set_params(self, g, params):
        detrend, start = params
        self._detrend[g] = detrend
        self._start[g] = start

    def _compute_params(self, g, X):
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
        params = kwparams(self.kwargs, "method")
        if self.method in [None, 'identity']:
            return IndentityTrend()
        elif self.method == 'linear':
            return FunctionTrend(fun=poly1)
        elif self.method == 'piecewise':
            return PiecewiseTrend(self.sp, **params)
        elif self.method == 'stepwise':
            return StepwiseTrend(self.sp, **params)
        elif self.method == 'global':
            return ConstantTrend(self.sp, **params)
        elif self.method in 'poly1':
            return FunctionTrend(fun=poly1)
        elif self.method in 'poly3':
            return FunctionTrend(fun=poly3)
        elif self.method == 'power':
            return FunctionTrend(fun=power1)
        else:
            raise ValueError(f"Unsupported detrend method {self.method}")
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
