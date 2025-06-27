
__all__ = [
    "ZeroForecaster",
]

from typing import Literal

import numpy as np

from .base import BaseForecaster
from stdlib.is_instance import is_instance


# ---------------------------------------------------------------------------
# ConstantForecaster
# ---------------------------------------------------------------------------

class ZeroForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    def __init__(self, mode: Literal[0, "zero", "mean", "median", "min", "max"]):
        super().__init__()

        assert is_instance(mode, Literal[0, "zero", "mean", "median", "min", "max"])

        self.mode = mode
        self._y_value = None

    def _fit(self, y: np.ndarray, X=None, fh=None):
        if self.mode == 0:
            self._y_value = 0
        elif self.mode == 'mean':
            self._y_value = y.mean()
        elif self.mode == 'median':
            self._y_value = np.median(y)
        elif self.mode == "min":
            self._y_value = y.min()
        elif self.mode == "max":
            self._y_value = y.mean()
        return self

    def _predict(self, fh, X=None):
        _y_shape = self._y.shape
        if len(_y_shape) == 1:
            y_pred = np.zeros(len(fh))
        else:
            y_pred = np.zeros((len(fh),) + _y_shape[1:])

        if self.const_value != 0:
            y_pred += self.const_value
        return y_pred

    def __repr__(self, **kwargs):
        if self.const_value == 0:
            return f"ZeroForecaster"
        else:
            return f"ZeroForecaster[value={self.const_value}]"

