# Used to remove heteroskedasticity (and trend)
#
# Other methods (sktime):
#
#   BoxCoxTransformer
#   LogTransformer
#   ExponentTransformer
#   SqrtTransformer
#
#   ScaledAsinhTransformer
#       Known as variance stabilizing transformation, Combined with an
#
#       sktime.forecasting.compose.TransformedTargetForecaster
#
#       can be useful in time series that exhibit spikes
#
#   SlopeTransform
# .

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.optimize as spo

from stdlib.mathx import isgt, islt, sq, sqrt
from .base import GroupsBaseEncoder

ARRAY = np.ndarray
INT_ARRAY = np.ndarray
FUNCTION = type(lambda x: None)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def period_diff(pi: pd.PeriodIndex, start: pd.Period) -> INT_ARRAY:
    diff = pi - start
    if len(diff) == 0:
        return np.array([])
    if isinstance(diff[0], int):
        return diff
    diff = np.array([e.n for e in diff], dtype=int)
    return diff


# y = a0 + a1 x
def poly1(x, a0, a1): return a1*x + a0


# y = a0 + a1 x + a2 x^2 + a3 x^3
def poly3(x, a0, a1, a2, a3): return ((a3*x + a2)*x * a1)*x + a0


# y = a0 + a1 x + a2 x^2 + a3 x^3 + a4 x^4 + a5 x^5
def poly5(x, a0, a1, a2, a3, a4, a5):
    return ((((a5*x + a4)*x * a3)*x + a2)*x + a1)*x + a0


# y = a0 + a1 x^a2
def power1(x, a0, a1, a2): return a0 + a1*(x**a2)


# y = a0 + a1 e^(a2 x)
def exp1(x, a0, a1, a2): return a0 + a1*np.exp(-a2*x)


# y = a0 + a1 log(x + a2)
def log1(x, a0, a1, a2): return a0 + a1*np.log(x+a2)


# ---------------------------------------------------------------------------

def select_points(x: ARRAY, y: ARRAY, upper=False) -> tuple[float, float, float, float]:
    lower = not upper
    n = len(y)
    x0 = float(x[0])
    y0 = float(y[0])
    xn = float(x[-1])
    yn = float(y[-1])
    swap = False

    i, j = 0, n-1
    while i < j:
        xi = float(x[i])
        yi = float(y[i])
        yt = y0 + (xi - x0) / (xn - x0) * (yn - y0)
        if upper and isgt(yi, yt) or lower and islt(yi, yt):
            x0 = x[i]
            y0 = y[i]
            i, j = 0, n - 1
            continue
        # end

        xj = float(x[j])
        yj = float(y[j])
        yt = y0 + (xj - x0) / (xn - x0) * (yn - y0)
        if upper and isgt(yj, yt) or lower and islt(yj, yt):
            xn = x[j]
            yn = y[j]
            i, j = 0, n - 1
            continue
        # end

        if swap:
            i += 1
        else:
            j -= 1
        swap = not swap
    # end
    return x0, y0, xn, yn


def compute_bounds(sp: int, x: ARRAY, y: ARRAY) -> tuple[ARRAY, ARRAY]:
    """
    Compute the bounds (min/max values) in each period of 'sp' slots.
    Note: the first period can be longer than 'sp' slots. This because
          it is preferred to have a 'complete period' at the end instead
          then at the begin of the arrays

    :param sp: period's length
    :param x: x values
    :param y: y values
    :return: a tuple (lower_bound, upper_bound)
    """
    n = len(x)
    m = max(1, n // sp)
    # s: offset used to have a complete seasonal period at the TS end
    s = n % sp

    lb = np.zeros((m, 2), dtype=float)
    ub = np.zeros((m, 2), dtype=float)

    j = 0
    for i in range(s, n, sp):
        # y in the period
        if i == s:
            # the first period can be longer than sp
            yp = y[0:i + sp]
        elif i < n-sp:
            yp = y[i:i + sp]
        else:
            yp = y[i:i + sp]

        arg = yp.argmin()
        lb[j, 0] = i + arg
        lb[j, 1] = yp[arg]

        arg = yp.argmax()
        ub[j, 0] = i + arg
        ub[j, 1] = yp[arg]

        j += 1
    # end
    return lb, ub


def set_begin_period(sp: int, x: ARRAY, bound: ARRAY) -> ARRAY:
    """
    Extract from x the value at the begin of each period.
    Note: the first value is not 'x[0]' but 'x[n%s]', this because
          it is preferred to have a complete period at the end of
          the array instead then at the  begin

    :param sp: period's length
    :param x: x values
    :param bound: where to save the values
    :return: bound itself
    """
    # correct x coordinate
    n = len(x)
    s = n % sp

    j = 0
    for i in range(s, n, sp):
        bound[j, 0] = x[i]
        j += 1
    return bound


# ---------------------------------------------------------------------------

def select_bound(xi: float, x: ARRAY, y: ARRAY) -> float:
    n = len(x)
    if xi < x[0]:
        return float(y[0])

    for i in range(n - 1):
        if x[i] <= xi < x[i + 1]:
            return float(y[i])

    if xi >= x[-1]:
        return float(y[-1])


def select_bounds(bound: ARRAY, x: ARRAY) -> ARRAY:
    n = len(x)
    yt = np.zeros_like(x, dtype=float)
    for i in range(n):
        yt[i] = select_bound(float(x[i]), bound[:, 0], bound[:, 1])
    return yt


# ---------------------------------------------------------------------------

def interpolate_bound(xi: float, x: ARRAY, y: ARRAY) -> float:
    n = len(x)
    if xi < x[0]:
        if x[1] != x[0]:
            return y[0] + (y[1] - y[0])*(xi - x[0])/(x[1] - x[0])
        else:
            return y[0]

    for i in range(n - 1):
        if x[i] <= xi < x[i + 1]:
            return y[i] + (y[i+1] - y[i])*(xi - x[i])/(x[i+1] - x[i])

    if xi >= x[-1]:
        return y[-2] + (y[-1] - y[-2])*(xi - x[-2])/(x[-1] - x[-2])


def interpolate_bounds(bound: ARRAY, x: ARRAY) -> ARRAY:
    n = len(x)
    yt = np.zeros_like(x, dtype=float)
    for i in range(n):
        yt[i] = interpolate_bound(float(x[i]), bound[:, 0], bound[:, 1])
    return yt


# ---------------------------------------------------------------------------

def fit_linear_bound(x: ARRAY, y: ARRAY, upper=False) -> FUNCTION:
    x0, y0, xn, yn = select_points(x, y, upper)
    # return lambda x: y0 + (yn - y0)*(x - x0)/(xn - x0)
    params = y0 - x0 * (yn - y0) / (xn - x0), (yn - y0) / (xn - x0)
    return lambda x: poly1(x, *params)


def fit_offset(fun: FUNCTION, x: ARRAY, y: ARRAY, upper=False) -> float:
    lower = not upper
    offset = 0.
    n = len(x)
    for i in range(n):
        yi = y[i]
        yt = fun(x[i]) + offset
        if upper and yi > yt or lower and yi < yt:
            offset += yi - yt
    return offset


def fit_offset_by_variance(fun: FUNCTION, x: ARRAY, y: ARRAY, k=1., upper=False) -> float:
    def sq(a): return a*a

    n = len(y)
    yt = fun(x)
    std = sqrt(sum(sq(y-yt))/n)
    return k*std if upper else -k*std


def fit_function(fun: FUNCTION, x: ARRAY, y: ARRAY) -> FUNCTION:
    params = spo.curve_fit(fun, x, y)[0]
    return lambda x: fun(x, *params)


def fit_bound(fun: FUNCTION, x: ARRAY, y: ARRAY, k=1, upper=False) -> FUNCTION:
    with np.errstate(all='ignore'):
        params = spo.curve_fit(fun, x, y)[0]
    if k is None:
        offset = fit_offset(lambda x: fun(x, *params), x, y, upper=upper)
    else:
        offset = fit_offset_by_variance(lambda x: fun(x, *params), x, y, k=k, upper=upper)
    return lambda x: fun(x, *params) + offset


# ---------------------------------------------------------------------------

def linear_interpolate(y: ARRAY, lb: ARRAY, ub: ARRAY) -> ARRAY:
    yt = (y - lb) / (ub - lb)
    return yt


def inverse_linear_interpolate(y: ARRAY, lb: ARRAY, ub: ARRAY) -> ARRAY:
    yt = y * (ub - lb) + lb
    return yt


# ---------------------------------------------------------------------------

def select_seasonal_values(sp: int, x: ARRAY, y: ARRAY, method='mean', centered=False) -> ARRAY:
    n = len(x)
    m = n//sp
    s = n % sp

    xy = np.zeros((m, 2), dtype=float)

    j = 0
    for i in range(s, n, sp):
        xy[j, 0] = i
        if method == 'mean':
            xy[j, 1] = y[i:i + sp].mean()
        elif method == 'median':
            xy[j, 1] = y[i:i + sp].median()
        elif method == 'min':
            xy[j, 1] = y[i:i + sp].min()
        elif method == 'max':
            xy[j, 1] = y[i:i + sp].max()
        else:
            raise ValueError(f"Unsupported method {method}")
        j += 1

    if centered:
        xy[:, 0] += sp//2
    return xy


# ---------------------------------------------------------------------------

def has_valid_pattern(y, tau: float) -> bool:
    """
    Check if the TS has a 'reasonable' behavior

    1) it subdivides the TS in 3 and next in 2 parts
    2) for each part computes the difference between the maximum and the minimum values

    if tau times the minimum difference is smaller than the maximum difference
    the TS is considered NOT valid

    This is used to understand if the TS contains spikes/outliers.

    :param y: the data to analyze
    :param tau: the factor (>1) used to compare the minimum vs the maximum difference
        between the minimum and the maximum values in a 'period'
    """
    if tau == 0:
        return True

    n = len(y)

    # split in 3 parts
    n3 = n//3
    nm = 2*n3
    y31 = y[:n3]
    y32 = y[n3:nm]
    y33 = y[nm:]
    y31diff, y32diff, y33diff = sorted([y31.max() - y31.min(), y32.max() - y32.min(), y33.max() - y33.min()])
    if tau*y31diff < y32diff or tau*y32diff < y33diff:
        return False

    # split in 2 and increase a little 'tau'
    tau *= 1.5
    n2 = n//2
    y21 = y[:n2]
    y22 = y[n2:]
    y21diff, y22diff = sorted([y21.max() - y21.min(),  y22.max() - y22.min()])
    if tau*y21diff < y22diff:
        return False

    return True


# ---------------------------------------------------------------------------
# MinMax
# ---------------------------------------------------------------------------

class MinMax:

    def __init__(self, sp, name="none"):
        self.name = name
        self.sp = sp if sp > 0 else 1
        self.err = .1
        self._lower_bound = None
        self._upper_bound = None
        self._invalid = False
        self._ymin = 0
        self._ymax = 0

    def fit(self, x: ARRAY, y: ARRAY):
        ...

    def transform(self, x: ARRAY, y: ARRAY) -> ARRAY:
        if self._invalid:
            return (y - self._ymin)/(self._ymax - self._ymin)
        else:
            yt = self._transform(x, y)
            yt[yt < 0] = 0.
            yt[yt > 1] = 1.
            return yt

    def inverse_transform(self, x: Optional[ARRAY], y: Optional[ARRAY]) -> ARRAY:
        if self._invalid:
            return self._ymin + y*(self._ymax - self._ymin)
        else:
            return self._inverse_transform(x, y)

    def _validate(self, y: ARRAY, lb: Optional[ARRAY], ub: Optional[ARRAY]):
        err = self.err
        if lb is None:
            self._invalid = True
            self._ymin = y.min()
            self._ymax = y.max()
            return

        yt = (y - lb) / (ub - lb)
        if yt.min() < (0. - err) or yt.max() > (1. + err):
            # (logging.getLogger(self.__class__.__name__)
            #  .warning(f'[{self.name}] Interpolation exceeds the range [0,1]: [{yt.min():.3},{yt.max():.3}]'))
            self._invalid = True
            self._ymin = y.min()
            self._ymax = y.max()
        else:
            self._invalid = False

    def _transform(self, x: ARRAY, y: ARRAY) -> ARRAY:
        ...

    def _inverse_transform(self, x: ARRAY, y: ARRAY) -> ARRAY:
        ...
# end


# ---------------------------------------------------------------------------
# IdentityMinMax
# ---------------------------------------------------------------------------

class IdentityMinMax(MinMax):
    """
    Doesn't apply any transformation
    """
    def __init__(self):
        super().__init__(0, "identity")

    def fit(self, x, y):
        return self

    def transform(self, x, y):
        return y

    def inverse_transform(self, x, y):
        return y
# end


# ---------------------------------------------------------------------------
# StepwiseMinMax
# ---------------------------------------------------------------------------

class StepwiseMinMax(MinMax):
    """
    The lower/upper bounds are considered constants inside the period.
    The last period is extended to +infinity
    """
    def __init__(self, sp):
        super().__init__(sp, "stepwise")

    def fit(self, x, y):
        lb, ub = compute_bounds(self.sp, x, y)
        self._lower_bound = set_begin_period(self.sp, x, lb)
        self._upper_bound = set_begin_period(self.sp, x, ub)
        return self

    def transform(self, x, y):
        lb = select_bounds(self._lower_bound, x)
        ub = select_bounds(self._upper_bound, x)
        ub[ub == lb] += 1

        # yt = (y - lb) / (ub - lb)
        # return yt
        return linear_interpolate(y, lb, ub)

    def inverse_transform(self, x, y):
        lb = select_bounds(self._lower_bound, x)
        ub = select_bounds(self._upper_bound, x)

        # yt = y * (ub - lb) + lb
        # return yt
        return inverse_linear_interpolate(y, lb, ub)
# end


# ---------------------------------------------------------------------------
# PiecewiseMinMax
# ---------------------------------------------------------------------------

class PiecewiseMinMax(MinMax):
    """
    The lower/upper bounds are computer as linear interpolation between
    the bound in the current period and the bound in the next period.
    The last period is extended to +infinity.
    """

    def __init__(self, sp):
        super().__init__(sp, "piecewise")

    def fit(self, x, y):
        lb, ub = compute_bounds(self.sp, x, y)
        self._lower_bound = lb
        self._upper_bound = ub

        # check if the transformation is valid
        # lb = interpolate_bounds(self._lower_bound, x)
        # ub = interpolate_bounds(self._upper_bound, x)
        # super()._validate(y, lb, ub)
        return self

    def _transform(self, x, y):
        lb = interpolate_bounds(self._lower_bound, x)
        ub = interpolate_bounds(self._upper_bound, x)
        ub[ub == lb] += 1

        # yt = (y - lb) / (ub - lb)
        # return yt
        return linear_interpolate(y, lb, ub)

    def _inverse_transform(self, x, y):
        lb = interpolate_bounds(self._lower_bound, x)
        ub = interpolate_bounds(self._upper_bound, x)

        # yt = y * (ub - lb) + lb
        # return yt
        return inverse_linear_interpolate(y, lb, ub)
# end


# ---------------------------------------------------------------------------
# LinearMinMaxInfo
# ---------------------------------------------------------------------------

class LinearMinMax(MinMax):
    """
    The lower/upper bounds are computed as linear interpolation of the
    minimum/maximum values in each period.
    Note: it is also implemented a 'trick' to check if the interpolation
    is valid: the minimum/maximum interpolated values can not be lass then
    -0.1 or greater than 1.1
    """

    def __init__(self, sp, tau=0.1):
        super().__init__(sp, "linear")
        self.tau = tau

    def fit(self, x, y):
        lb, ub = compute_bounds(self.sp, x, y)
        self._lower_bound = fit_linear_bound(lb[:, 0], lb[:, 1], upper=False)
        self._upper_bound = fit_linear_bound(ub[:, 0], ub[:, 1], upper=True)

        # check if the transformation is valid
        # lb = self._lower_bound(x)
        # ub = self._upper_bound(x)
        # super()._validate(y, lb, ub)
        return self

    def _transform(self, x, y):
        lb = self._lower_bound(x)
        ub = self._upper_bound(x)
        ub[ub == lb] += 1

        # yt = (y - lb) / (ub - lb)
        # return yt
        return linear_interpolate(y, lb, ub)

    def _inverse_transform(self, x, y):
        lb = self._lower_bound(x)
        ub = self._upper_bound(x)

        # yt = y * (ub - lb) + lb
        # return yt
        return inverse_linear_interpolate(y, lb, ub)
# end


# ---------------------------------------------------------------------------
# RatioMinMax
# ---------------------------------------------------------------------------

class RatioMinMax(MinMax):
    """
    Scale the y values in such way that the minimum value
    is mapped to 'min_ratio' and the maximum value to 'max_ratio'
    There are two 'standard' values:

        - (0.0, 1.0)
        - (0.2, 0.8)

    The second one is used to support TS with a 'trend' but the
    TS model used is not able to support it (for example LSTM/GRU/RNN,
    limited in the range [0,1) ).
    """

    def __init__(self, ratio=(0, 1)):
        super().__init__(0, "ratio")
        if isinstance(ratio, (int, float)):
            min_ratio, max_ratio = 0, ratio
        else:
            min_ratio, max_ratio = ratio

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self._lower_bound: ARRAY = np.zeros(0)
        self._upper_bound: ARRAY = np.zeros(0)

    def fit(self, x: ARRAY, y: ARRAY):
        min_ratio = self.min_ratio
        miny = min(y)
        if min_ratio is None:
            self._lower_bound = 0
        elif min_ratio == 0:
            self._lower_bound = miny
        elif min_ratio < 1:
            self._lower_bound = miny - miny * min_ratio
        else:
            self._lower_bound = miny - miny / min_ratio

        max_ratio = self.max_ratio
        maxy = max(y)
        if max_ratio == 1:
            self._upper_bound = maxy
        elif max_ratio < 1:
            self._upper_bound = maxy + maxy / max_ratio
        else:
            self._upper_bound = maxy + maxy * max_ratio

    def transform(self, x, y):
        lb = self._lower_bound
        ub = self._upper_bound

        # yt = (y - lb) / (ub - lb)
        # return yt
        return linear_interpolate(y, lb, ub)

    def inverse_transform(self, x, y):
        lb = self._lower_bound
        ub = self._upper_bound

        # yt = y * (ub - lb) + lb
        # return yt
        return inverse_linear_interpolate(y, lb, ub)
# end


# ---------------------------------------------------------------------------
# FunctionMinMax
# ---------------------------------------------------------------------------
# scipy.optimize.curve_fit  (curve fitting)
# numpy.polyfit             (polynomial fitting)
#

class FunctionMinMax(MinMax):
    """
    The lower/upper bounds are computed as functional interpolation of the
    minimum/maximum values in each period, based on the parametric function
    specified.

    The parametric functions available are:

        'poly1'     y = a0 + a1 x
        'poly3'     y = a0 + a1 x + a2 x^2 + a3 x^3
        'poly5'     y = a0 + a1 x + a2 x^2 + a3 x^3 + a4 x^4 + a5 x^5
        'power1'    y = a0 + a1 x^a2
        'exp1'      y = a0 + a1 e^(a2 x)
        'log1'      y = a0 + a1 log(x + a2)

    Note: it is also implemented a 'trick' to check if the interpolation
    is valid: the minimum/maximum interpolated values can not be lass then
    -0.1 or greater than 1.1
    """

    def __init__(self, fun, sp):
        super().__init__(sp, f"function[{fun}]")
        self.fun = fun

    def fit(self, x, y):
        try:
            lb, ub = compute_bounds(self.sp, x, y)
            self._lower_bound = fit_bound(self.fun, lb[:, 0], lb[:, 1], upper=False)
            self._upper_bound = fit_bound(self.fun, ub[:, 0], ub[:, 1], upper=True)

            # check if the transformation is valid
            # lb = self._lower_bound(x)
            # ub = self._upper_bound(x)
            # super()._validate(y, lb, ub)
        except Exception as e:
            logging.getLogger('FunctionMinMax').warning('Unable to fit the function')
            super()._validate(y, None, None)
        return self

    def _transform(self, x, y):
        lb = self._lower_bound(x)
        ub = self._upper_bound(x)
        ub[ub == lb] += 1.

        # yt = (y - lb)/(ub - lb)
        # return yt
        return linear_interpolate(y, lb, ub)

    def _inverse_transform(self, x, y):
        ub = self._lower_bound(x)
        lb = self._upper_bound(x)

        # yt = y*(ub - lb) + lb
        # return yt
        return inverse_linear_interpolate(y, lb, ub)
# end


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------
# supported methods
#   None            identity
#   identity        identity
#   global          single min/max values (global)  (same as  (0, 1))
#   max             ratio  (global)                 (same as  (0, max)
#   (min, max)      ratio  (global)
#   linear          linear (global)
#
#   piecewise       linear   (by seasonality)
#   stepwise        constant (by seasonality)
#
#   poly1           polynomial  interpolation       a0 + a1 x
#   poly3           polynomial  interpolation       a0 + a1 x + a2 x^2 + a3 x^3
#   power           exponential interpolation       a0 + a1 x^a2
#   exp             exponential interpolation       a0 + a1 e^(a2 x)
#

class ConfigurableMinMaxScaler(GroupsBaseEncoder):

    def __init__(self, columns=None, feature_range=(0, 1), *,
                 method=None, sp=None, tau=0.,
                 groups=None, copy=True):
        """

        :param columns: columns where to apply the minimum maximum scaling.
                If None, it is applied to all columns
        :param feature_range: in format '(min_val, max_val)', mapped values range
        :param method: method to use to compute the bounds
                Available methods:
                    None              identity
                    'identity'        identity
                    'global'          single min/max values (global)
                    float             ratio  (global)
                    (float, float)    ratio  (global)
                    'linear'          linear (global)
                    'piecewise'       linear   (by seasonality)
                    'stepwise'        constant (by seasonality)
                    'poly1'           polynomial  interpolation       a0 + a1 x
                    'poly3'           polynomial  interpolation       a0 + a1 x + a2 x^2 + a3 x^3
                    'power'           exponential interpolation       a0 + a1 x^a2
                    'exp'             exponential interpolation
        :param sp: seasonality period
        :param tau: if > 0, the TS is validated: the TS is valid if the maximum difference between
            the maximum and the minimum values in a period is less that 'tau' times the minimum
            difference between the maximum and the minimum values in a period
        :param groups: columns used to identify the TS in a multi-TS dataset
                If None, it is used the MultiIndex
        :param copy:
        """
        super().__init__(columns=columns, groups=groups, copy=copy)
        self.feature_range: tuple[float, float] = feature_range
        self.method: str = method
        self.sp: int = sp  # seasonality period
        self.tau: float = tau

        # MinMax scaler for each group
        self._minmax: dict[tuple, MinMax] = {}
        self._start = {}

    def _get_params(self, g):
        return self._minmax[g], self._start[g]

    def _set_params(self, g, params):
        mmi, start = params
        self._minmax[g] = mmi
        self._start[g] = start

    def _compute_params(self, g, X):
        n = len(X)
        minmax = {}
        start = X.index[0]

        xc = np.arange(n)
        for col in self._get_columns(X):
            yc = X[col].to_numpy()

            if not has_valid_pattern(yc, self.tau):
                logging.getLogger("MinMaxScaler").warning(f"TS column {col} doesn't present a valid pattern")
                mmi = RatioMinMax()
            else:
                mmi = self._create_mmi()

            # set the column name
            mmi.name = col
            mmi.fit(xc, yc)
            minmax[col] = mmi
        # end
        return minmax, start

    def _create_mmi(self):
        sp = self.sp
        method = self.method

        if method not in [None, 'identity', 'global'] and sp in [0, None]:
            print(f"[WARNING: specified 'method' '{method}', but 'sp' is not specified. Forced to 12 (months)")
            sp = 12

        # no scaling
        if method in [None, 'identity']:
            return IdentityMinMax()

        # global or (min,max)
        elif method == "global" or isinstance(method, (int, float, list, tuple)):
            return RatioMinMax()
        elif isinstance(method, (int, float, list, tuple)):
            ratio = method
            return RatioMinMax(ratio)

        # linear
        elif method == 'linear':
            return LinearMinMax(sp)
        elif method == 'piecewise':
            return PiecewiseMinMax(sp)
        elif method == 'stepwise':
            return StepwiseMinMax(sp)

        # function based
        elif method == 'poly1':
            return FunctionMinMax(poly1, sp)
        elif method == 'poly3':
            return FunctionMinMax(poly3, sp)
        elif method == 'power':
            return FunctionMinMax(power1, sp)
        elif method == 'exp':
            return FunctionMinMax(exp1, sp)

        # error
        else:
            raise ValueError(f"Unsupported method '{method}'/{sp}")

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
            yt = ymin + (ymax - ymin) * yt
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
            yt = (yc - ymin) / (ymax - ymin)
            yt = minmax[col].inverse_transform(xi, yt)
            X[col] = yt
        # end
        return X

    # -----------------------------------------------------------------------
    # Extra features
    # -----------------------------------------------------------------------
# end


class MinMaxScaler(GroupsBaseEncoder):

    def __init__(self, columns=None, feature_range=(0, 1), *,
                 clip=False,
                 threshold=0,
                 groups=None, copy=True):
        """
        As the scikit-learn MinMaxScaler but with the possibility to remove
        some lowest/highest values
        :param columns:
        :param feature_range:
        :param clip:
        :param threshold:
        :param groups:
        :param copy:
        """
        super().__init__(columns=columns, groups=groups, copy=copy)
        self.feature_range = feature_range
        self.threshold = threshold
        self.clip = clip

        self._min = {}
        self._max = {}
    # end

    def _get_params(self, g):
        return self._min[g], self._max[g]

    def _set_params(self, g, params):
        cmin, cmax = params
        self._min[g] = cmin
        self._max[g] = cmax

    def _compute_params(self, g, X):
        threshold = self.threshold
        n = len(X)
        start = threshold if threshold > 1 else int(threshold*n)

        cmin = {}
        cmax = {}
        for col in self._get_columns(X):
            if start != 0:
                end = n - start
                xc = np.sort(X[col])
                xc = xc[start:end]
            else:
                xc = X[col]

            xcmin = xc.min()
            xcmax = xc.max()

            cmin[col] = xcmin
            cmax[col] = xcmax
        # end
        return cmin, cmax

    def _apply_transform(self, X, params):
        cmin, cmax = params
        minv, maxv = self.feature_range
        clip = self.clip
        deltav = maxv - minv

        Xt = X.copy()
        for col in self._get_columns(X):
            xcmin = cmin[col]
            xcmax = cmax[col]

            xc = X[col]

            if xcmin < xcmax:
                xt = minv + (xc - xcmin)/(xcmax - xcmin)*deltav
            else:
                xt = minv + (xc - xc)

            if clip:
                xt[xt < minv] = minv
                xt[xt > maxv] = maxv

            Xt[col] = xt

        return Xt

    def _apply_inverse_transform(self, X, params):
        cmin, cmax = params
        minv, maxv = self.feature_range
        deltav = maxv - minv

        Xt = X.copy()
        for col in self._get_columns(X):
            xcmin = cmin[col]
            xcmax = cmax[col]

            xc = X[col]

            xt = xcmin + (xc - minv)/deltav*(xcmax - xcmin)

            Xt[col] = xt

        return Xt
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
