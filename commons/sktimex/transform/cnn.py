import numpy as np
from typing import Optional

from .base import ModelTrainTransform, ModelPredictTransform
from ..lag import resolve_lags, lmax


# ---------------------------------------------------------------------------
# CNNTrainTransform
# CNNPredictTransform
# ---------------------------------------------------------------------------
# N, Channels, Channel_Length
# N, Data,     Channel_Length

# Difference between CNNTrainTransform AND CNNFlatTrainTransform
# --------------------------------------------------------------
# The only difference is in y:  y[n, my, s]  vs  y[n, my]
# and how y is initialized

class CNNTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

        xlags = self.xlags
        ylags = self.ylags
        assert len(xlags) == 0 or xlags == ylags, "Supported only [0, n], [n, n]"
    # end

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().transform(X, y)

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

        yt = np.zeros((n, my*st), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in range(st):
                k = tlags[j]
                yt[i, c:c + my] = y[i + t + k]
                c += my
            # end
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class CNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        X, fh = super().transform(X, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        y = self.yh

        if fh == 0: fh = len(X)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        mu = my * st

        Xt = np.zeros((1, mt, sy), dtype=y.dtype)
        yp = np.zeros((fh, mu), dtype=y.dtype)

        self.Xp = X
        self.yp = yp
        self.Xt = Xt

        return yp
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
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

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
    # end
# end

