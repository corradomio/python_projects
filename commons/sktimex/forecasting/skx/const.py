
__all__ = [
    "ConstantForecaster",
    "ZeroForecaster",
]

from typing import Literal

import numpy as np

from ..base import BaseForecaster
from ...utils import to_numpy, to_data, is_instance


# ---------------------------------------------------------------------------
# ConstantForecaster
# ---------------------------------------------------------------------------

class ConstantForecaster(BaseForecaster):

    # _tags = {
    #     "capability:exogenous": True,
    #     "requires-fh-in-fit": False,
    #     "capability:missing_values": True
    # }

    def __init__(self, mode: Literal[0, "zero", "mean", "median", "min", "max"] = 0):
        super().__init__()

        assert is_instance(mode, Literal[0, "0", "zero", "mean", "median", "min", "max"])

        self.mode = mode
        self._y_pred = None
    # end

    def _fit(self, y, X, fh):
        y = to_numpy(y)

        if self.mode in ["zero", "zeros", "0", 0]:
            if len(y.shape) == 1:
                zero = np.zeros(1)
            else:
                zero = np.zeros(y.shape[1:])
            self._y_pred = zero
        elif self.mode == 'mean':
            self._y_pred = y.mean(axis=0)
        elif self.mode == 'median':
            self._y_pred = np.median(y, axis=0)
        elif self.mode == "min":
            self._y_pred = y.min(axis=0)
        elif self.mode == "max":
            self._y_pred = y.mean(axis=0)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
        return self

    def _predict(self, fh, X):
        super()._predict(fh, X)
        y_shape = (len(fh),) + self._y_pred.shape
        y_pred = np.zeros(y_shape)
        y_pred[:] = self._y_pred

        return to_data(y_pred, self._y, self.cutoff, fh)

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self, **kwargs):
        if self._y_pred == 0:
            return f"ConstantForecaster"
        else:
            return f"ConstantForecaster[{self.mode}={self._y_pred}]"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params_list = [
            {},
            {"mode": "zero"},
            {"mode": "mean"},
            {"mode": "median"},
            {"mode": "min"},
            {"mode": "max"},
        ]
        return params_list

# end

ZeroForecaster=ConstantForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
