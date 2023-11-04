from typing import Union, Any, Optional

import numpy as np
import pandas as pd

# DON'T REMOVE!!!!
# They are used in other modules to avoid the direct dependency with 'stdlib'
from stdlib import NoneType, kwval, dict_del, import_from, qualified_name, lrange   # DON'T REMOVE!!!!
# DON'T REMOVE!!!!

from sktime.forecasting.base import ForecastingHorizon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[NoneType, int, list[int], np.ndarray, ForecastingHorizon]
PD_TYPES = Union[NoneType, pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def lmax(l: list) -> int:
    if l is None or len(l) == 0:
        return 0
    else:
        return max(l)


def to_matrix(data: Union[NoneType, pd.Series, pd.DataFrame, np.ndarray], dtype=np.float32) -> Optional[np.ndarray]:
    if data is None:
        return None
    if isinstance(data, pd.Series):
        data = data.to_numpy().astype(dtype).reshape((-1, 1))
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy().astype(dtype)
    elif len(data.shape) == 1:
        assert isinstance(data, np.ndarray)
        data = data.astype(dtype).reshape((-1, 1))
    else:
        assert isinstance(data, np.ndarray)
        data = data.astype(dtype)
    return data


def fh_range(n: int) -> ForecastingHorizon:
    return ForecastingHorizon(list(range(1, n+1)))


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
