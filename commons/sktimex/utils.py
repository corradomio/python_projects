from typing import Union, Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from stdlib import NoneType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[NoneType, int, list[int], np.ndarray, ForecastingHorizon]
PD_TYPES = Union[NoneType, pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def fh_range(n: int) -> ForecastingHorizon:
    return ForecastingHorizon(list(range(1, n+1)))


def import_from(qname: str) -> Any:
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz


def dict_del(d: dict, keys: Union[str, list[str]]) -> dict:
    d = {} | d
    if isinstance(keys, str):
        keys = list[keys]
    for k in keys:
        if k in d:
            del d[k]
    return d


def kwval(kwargs: dict[Union[str, tuple], Any], key: Union[str, tuple], defval: Any = None) -> Any:
    if key not in kwargs:
        return defval

    def _tobool(s: str) -> bool:
        if s in [0, False, '', 'f', 'false', 'F', 'False', 'FALSE', 'off', 'no', 'close']:
            return False
        if s in [1, True, 't', 'true', 'T', 'True', 'TRUE', 'on', 'yes', 'open']:
            return True
        else:
            raise ValueError(f"Unsupported boolean value '{s}'")

    val = kwargs[key]
    if not isinstance(defval, str) and isinstance(val, str):
        if defval is None:
            return val
        if isinstance(defval, bool):
            return _tobool(val)
        if isinstance(defval, int):
            return int(val)
        if isinstance(defval, float):
            return float(val)
        else:
            raise ValueError(f"Unsupported conversion from str to '{type(defval)}'")
    return val


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
