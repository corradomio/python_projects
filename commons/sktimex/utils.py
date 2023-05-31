from typing import Union, Optional
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[type(None), int, list[int], np.ndarray, ForecastingHorizon]
PD_TYPES = Union[type(None), pd.Series, pd.DataFrame]

