from typing import Union, Optional

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from numpy import ndarray

from .lag import resolve_lag, LagSlots
from .stdlib import NoneType, import_from

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_TYPES = (NoneType, pd.DataFrame, np.ndarray)
TARGET_TYPES = (NoneType, pd.DataFrame, pd.Series, np.ndarray)
FH_TYPES = (NoneType, int, list, ndarray, pd.Index, pd.TimedeltaIndex, pd.Timedelta, skf.ForecastingHorizon)
X_EMPTY = np.zeros((0, 0))


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------

def format_data(y: np.ndarray, slots: LagSlots = None,
                X: Optional[np.ndarray] = None) \
    -> (np.ndarray, np.ndarray):

    s = len(slots)

    lx = len(slots.input_slots)
    ly = len(slots.target_slots)
    k = y.shape[0]

    if X is None:
        X = X_EMPTY.reshape((k, 0))

    n, m = X.shape
    assert s < n == k

    nt = n - s
    mt = lx*m + ly
    tx = np.zeros((nt, mt))
    ty = np.zeros(nt)

    for i in range(nt):
        c = 0
        for j in reversed(slots.input_slots):
            tx[i, c:c+m] = X[s+i-j]
            c += m
        for j in reversed(slots.target_slots):
            tx[i, c] = y[s+i-j]
            c += 1
        ty[i] = y[s+i]
    # end

    return tx, ty
# end


def format_single(y: np.ndarray, slots: LagSlots,
                  X: Optional[np.ndarray] = None,
                  start: int = 0,
                  tx: np.ndarray = None):

    s = len(slots)
    assert start >= s

    if X is None:
        X = X_EMPTY.reshape((start+1, 0))

    n, m = X.shape

    lx = len(slots.input_slots)
    ly = len(slots.target_slots)

    mt = lx*m + ly
    if tx is None:
        tx = np.zeros((1, mt))

    c = 0
    i = 0
    for j in reversed(slots.input_slots):
        tx[i, c:c+m] = X[s+i-j]
        c += m
    for j in reversed(slots.target_slots):
        tx[i, c] = y[s+i-j]
        c += 1

    return tx
# end


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------

#
# We suppose that the dataset is ALREADY normalized.
# the ONLY information is to know the name of the target column'
#

class LinearModel:

    def __init__(self,
                 class_name: str,
                 lag: Union[int, list, tuple, dict],
                 **kwargs):

        self._lag = lag
        model_class = import_from(class_name)
        self._model = model_class(**kwargs)
    # end

    def fit(self, y: TARGET_TYPES, X: INPUT_TYPES = None, fh: FH_TYPES = None):
        y, X, fh = self._validate_data(y, X, fh)
        slots = resolve_lag(self._lag)
        Xm, ym = format_data(X=X, y=y, slots=slots)
        self._model.fit(Xm, ym)
        return self

    def predict(self, fh: FH_TYPES, X: INPUT_TYPES = None, y: TARGET_TYPES = None) -> pd.Series:
        fh, y, X = self._validate_data(fh, y, X)
        slots = resolve_lag(self._lag)
        Xlm, ylm = format_data(X=X, y=y, slots=slots)
        return y

    def score(self, fh: FH_TYPES, X: INPUT_TYPES, y: TARGET_TYPES) -> dict[str, float]:
        fh, y, X = self._validate_data(fh, y, X)
        return {}

    def _validate_data(self, y=None, X=None, fh=None):

        if X is not None:
            assert isinstance(X, INPUT_TYPES)
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 2

        if y is not None:
            assert isinstance(y, TARGET_TYPES)
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.to_numpy()
            assert isinstance(y, np.ndarray)
            assert len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1
            if len(y.shape) == 2:
                y = y.reshape(-1)

        if fh is not None:
            assert isinstance(fh, FH_TYPES)
            if not isinstance(fh, skf.ForecastingHorizon):
                fh = skf.ForecastingHorizon(fh)

        return y, X, fh
    # end
# end
