from typing import Union, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .plotting import *


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# DON'T REMOVE!!!!
# They are used in other modules to avoid the direct dependency with 'stdlib'
from stdlib import kwexclude, kwval, kwmerge
from stdlib import lrange, import_from, qualified_name, as_dict, as_list
from stdlib import mul_, is_instance, name_of, ns_of
# DON'T REMOVE!!!!


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

NoneType = type(None)
RangeType = type(range(0))
CollectionType = (list, tuple)
FunctionType = type(lambda x: x)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[NoneType, int, list[int], np.ndarray, ForecastingHorizon]
PD_TYPES = Union[NoneType, pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# method_of
# ---------------------------------------------------------------------------

def method_of(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


# ---------------------------------------------------------------------------
# clear_yX
# ---------------------------------------------------------------------------

def _has_xyfh(model):
    return hasattr(model, "_X") and hasattr(model, "_y") and hasattr(model, "_fh")


def clear_yX(model):
    if isinstance(model, list):
        estimators_list = model
        for estimator in estimators_list:
            clear_yX(estimator)

    elif isinstance(model, dict):
        estimators_dict = model
        for key in estimators_dict:
            clear_yX(estimators_dict[key])

    elif _has_xyfh(model):
        model._X = None
        model._y = None

        for attr in [
            "estimators", "_estimators", "estimators_",
            "forecasters", "_forecasters", "forecasters_",
            "estimator", "_estimator", "estimator_",
            "forecaster", "_forecaster", "forecaster_",
        ]:
            if hasattr(model, attr):
                clear_yX(getattr(model, attr))
    return
# end


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_matrix(data: Union[NoneType, pd.Series, pd.DataFrame, np.ndarray], dtype=np.float32) -> Optional[np.ndarray]:
    if data is None:
        pass
    elif isinstance(data, pd.Series):
        data = data.to_numpy().astype(dtype).reshape((-1, 1))
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy().astype(dtype)
    elif data.ndim == 1:
        data = data.astype(dtype).reshape((-1, 1))
    elif isinstance(data, np.ndarray) and dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    elif isinstance(data, np.ndarray):
        pass
    return data


def to_matrices(*dlist) -> list[np.ndarray]:
    mlist = []
    for d in dlist:
        mlist.append(to_matrix(d))
    return mlist


# def fh_range(n: int) -> ForecastingHorizon:
#     return ForecastingHorizon(list(range(1, n+1)))


def make_lags(lags, current):
    if current is None:
        return lags
    if isinstance(lags, int):
        return 0, lags, current
    elif len(lags) == 1:
        return 0, lags[0], current
    elif isinstance(lags, list):
        return tuple(lags) + (current,)
    else:
        return tuple(lags) + (current,)

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
