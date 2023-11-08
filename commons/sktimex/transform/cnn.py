import numpy as np
from typing import Optional

from .base import ModelTrainTransform, ModelPredictTransform
from ..lags import resolve_lags, lmax


# ---------------------------------------------------------------------------
# CNNTrainTransform
# CNNPredictTransform
# ---------------------------------------------------------------------------

class CNNTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None, flatten=False):
        # lags is an alternative to slots
        super().__init__(
            slots=lags if lags is not None else slots,
            tlags=tlags)

        self.flatten = flatten
        xlags = self.xlags
        ylags = self.ylags
        assert len(xlags) == 0 or xlags == ylags, "Supported only [0, n], [n, n]"
    # end

    def transform(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().transform(y=y, X=X)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        t = len(self.slots)
        v = lmax(tlags)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my

        n = y.shape[0] - (t + v)

        Xt = np.zeros((n, mt, sy), dtype=y.dtype)

        for i in range(n):
            for j in range(sy):
                k = ylags[sy - 1 - j]   # reversed
                Xt[i, 0:my, j] = y[i + t - k]
            # end
        # end

        for i in range(n):
            for j in range(sx):
                k = xlags[sx - 1 - j]   # reversed
                Xt[i, my:, j] = X[i + t - k]
            # end
        # end

        yt = np.zeros((n, my, st), dtype=y.dtype)

        for i in range(n):
            # pass
            for j in range(st):
                k = tlags[j]
                yt[i, :, j] = y[i + t + k]
                # pass
            # end
        # end

        if self.flatten:
            yt = yt.reshape((yt.shape[0], -1))

        return Xt, yt
    # end
# end


class CNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None, flatten=False):
        # lags is an alternative to slots
        super().__init__(
            slots=lags if lags is not None else slots,
            tlags=tlags)

        self.flatten = flatten

    def transform(self, fh: int = 0, X: Optional[np.ndarray] = None) -> np.ndarray:
        fh, X = super().transform(fh, X)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags

        sx = len(xlags)
        sy = len(ylags)

        y = self.yh

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        #

        Xt = np.zeros((1, mt, sy), dtype=y.dtype)
        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xp = X
        self.yp = yp
        self.Xt = Xt

        return self.to_pandas(yp)
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i, 0]

    def step(self, i) -> np.ndarray:
        atx = self._atx
        aty = self._aty

        xlags = self.xlags if self.Xh is not None else []
        ylags = self.ylags

        sx = len(xlags)
        sy = len(ylags)

        Xt = self.Xt
        my = self.yh.shape[1]

        for j in range(sy):
            k = ylags[sy - 1 - j]   # reversed
            Xt[0, 0:my, j] = aty(i - k)
        # end

        for j in range(sx):
            k = xlags[sx - 1 - j]   # reversed
            Xt[0, my:, j] = atx(i - k)
        # end

        return Xt

    def update(self, i, y_pred, t=None):
        return super().update(i, y_pred, t)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
