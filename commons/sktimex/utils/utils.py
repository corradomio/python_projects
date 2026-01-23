from typing import Union, Optional

import numpy as np
import pandas as pd
import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon



# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

NoneType = type(None)
RangeType = type(range(0))
CollectionType = (list, tuple)
FunctionType = type(lambda x: x)

PD_TYPES = Union[NoneType, pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKLEARN_NAMESPACES = ["sklearn.", "sklearnx.", "catboost.", "lightgbm.", "xgboost."]
SKTIME_NAMESPACES = ["sktime.", "sktimex.", "sktimexnn.", "sktimext."]

def starts_with(name: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if name.startswith(p):
            return True
    return False
# end

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


def clear_yX(model, recursive=False):
    if isinstance(model, list):
        estimators_list = model
        for estimator in estimators_list:
            clear_yX(estimator, recursive=recursive)

    elif isinstance(model, dict):
        estimators_dict = model
        for key in estimators_dict:
            clear_yX(estimators_dict[key], recursive=recursive)

    elif _has_xyfh(model):
        model._X = None
        model._y = None

        if not recursive:
            return

        for attr in [
            "estimators", "_estimators", "estimators_",
            "forecasters", "_forecasters", "forecasters_",
            "estimator", "_estimator", "estimator_",
            "forecaster", "_forecaster", "forecaster_",
        ]:
            if hasattr(model, attr):
                clear_yX(getattr(model, attr), recursive=recursive)
    return
# end


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dtype_of(X, y):
    if X is None and y is None:
        return np.float64
    if X is None and y is not None:
        return y.dtype
    if X is not None and y is None:
        return X.dtype
    return np.promote_types(X.dtype, y.dtype)


# ---------------------------------------------------------------------------
# numpy/pandas conversions
# ---------------------------------------------------------------------------

# def is_fh_range(fh: ForecastingHorizon) -> bool:
#     assert isinstance(fh, ForecastingHorizon)
#     assert fh.is_relative
#     n = len(fh)
#     fh0 = fh[0]
#     fhn = fh[-1]
#     return fh0 == 1 and n == (fhn - fh0 + 1)


# def to_numpy(data: Union[NoneType, pd.Series, pd.DataFrame, np.ndarray], *,
#              dtype=None,
#              matrix=False) -> Optional[np.ndarray]:
#     assert isinstance(data, (NoneType, pd.Series, pd.DataFrame, np.ndarray))
#
#     if data is None:
#         pass
#     elif isinstance(data, pd.Series):
#         data = data.to_numpy()
#     elif isinstance(data, pd.DataFrame):
#         data = data.to_numpy()
#     elif isinstance(data, np.ndarray):
#         pass
#
#     if matrix and len(data.shape) == 1:
#         data = data.reshape((-1, 1))
#
#     if data is not None and dtype is not None and data.dtype != dtype:
#         data = data.astype(dtype)
#     return data


def to_data(data, y, cutoff, fh: ForecastingHorizon):
    assert isinstance(data, np.ndarray)
    assert isinstance(y, (pd.Series, pd.DataFrame, np.ndarray))
    # cutoff is an ?Index of length 1
    cutoff = cutoff[0]

    if isinstance(y, np.ndarray):
        return data

    index = to_index(y, cutoff, fh)

    if isinstance(y, pd.Series):
        assert len(data.shape) == 1 or len(data.shape) == 2 and data.shape[1] == 1
        ser = pd.Series(data=data.reshape(-1), index=index, name=y.name)
        return ser
    elif isinstance(y, pd.DataFrame):
        df = pd.DataFrame(data=data, index=index, columns=y.columns)
        return df
    else:
        raise ValueError(f"Unsupported type: {type(y)}")
#  end


def to_index(y, cutoff, fh: ForecastingHorizon):
    if fh.is_relative:
        fh = fh.to_absolute(cutoff)

    y_index = y.index
    if isinstance(y_index, pd.RangeIndex):
        index = pd.Index(fh.to_numpy())
    elif isinstance(y_index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(fh)
    elif isinstance(y_index, pd.PeriodIndex):
        index = pd.PeriodIndex(fh)
    elif isinstance(y_index, pd.Index):
        index = pd.Index(fh.to_numpy())
    else:
        raise ValueError(f"Unsupported index type: {type(y_index)}")

    return index
# end


# def to_relative(fh: ForecastingHorizon, cutoff) -> ForecastingHorizon:
#     if fh.is_relative:
#         return fh
#     else:
#         return fh.to_relative(cutoff)

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
