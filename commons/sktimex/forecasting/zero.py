
__all__ = [
    "ZeroForecaster",
]


import numpy as np

from .base import BaseForecaster
from ..utils import PD_TYPES


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

    def __init__(self, const_value=0):
        super().__init__()
        self.const_value = const_value

    def _fit(self, y, X=None, fh=None):
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

