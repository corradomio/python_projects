from typing import Optional

import numpy as np

from ..lags import LagSlots, resolve_lags, resolve_tlags
from ..utils import NoneType


# ---------------------------------------------------------------------------
#   ModelTransform
#       ModelTrainTransform
#       ModePredictTransform
# ---------------------------------------------------------------------------

class ModelTransform:

    def _check_X_y(self, X, y=None):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, (NoneType, np.ndarray))
# end


class ModelTrainTransform(ModelTransform):

    def __init__(self, slots, tlags=(0,)):
        if isinstance(slots, (list, tuple, dict)):
            slots = resolve_lags(slots)
        if isinstance(tlags, int):
            tlags = list(range(tlags))

        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list))

        self.slots = slots
        self.xlags: list = slots.input
        self.ylags: list = slots.target
        self.tlags: list = list(tlags)
    # end

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        self._check_X_y(X, y)
        return self
    # end

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_X_y(X, y)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        return X, y
    # end

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
# end


class ModelPredictTransform(ModelTransform):

    def __init__(self, slots, tlags=(0,)):
        if isinstance(slots, (list, tuple, dict)):
            slots = resolve_lags(slots)
        if isinstance(tlags, int):
            tlags = resolve_tlags(tlags)

        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list)), f"Parameter tlags not of type list|tuple: {tlags}"

        self.slots = slots
        self.xlags = slots.input
        self.ylags = slots.target
        self.tlags = tlags

        self.Xh = None  # X history
        self.yh = None  # y history

        self.Xt = None  # X transform
        self.yt = None  # y transform

        self.Xp = None  # X prediction
        self.yp = None  # y prediction
    # end

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            pass
        elif len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return self
    # end

    def transform(self, X: np.ndarray, fh: int):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(fh, int)
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if X is not None and fh == 0:
            fh = len(X)

        return X, fh
    # end
# end
