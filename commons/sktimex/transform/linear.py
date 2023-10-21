import numpy as np

from .base import ModelTrainTransform, ModelPredictTransform
from ..lags import resolve_lags, lmax


# ---------------------------------------------------------------------------
# LinearTrainTransform
# LinearPredictTransform
# ---------------------------------------------------------------------------
# (X, y, xslot, yslots) -> Xt, yt
#
# back_step
#   y[-1]             -> y[0]
#   y[-1],X[-1]       -> y[0]
#   y[-1],X[-1],X[0]  -> y[0]
#

class LinearTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)
        self.Xh = None
        self.yh = None

    def fit(self, X, y):
        super().fit(X, y)
        self.Xh = X
        self.yh = y
        return self

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().transform(X, y)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        t = len(self.slots)

        s = lmax(tlags)
        r = s + t

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = sx * mx + sy * my
        mu = st * my
        n = y.shape[0] - r

        Xt = np.zeros((n, mt), dtype=y.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in reversed(ylags):
                Xt[i, c:c + my] = y[t + i - j]
                c += my
            for j in reversed(xlags):
                Xt[i, c:c + mx] = X[t + i - j]
                c += mx

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[t + i + j]
                c += my
        # end

        return Xt, yt
    # end

    # def fit_transform(self, X, y):
    #     return self.fit(X, y).transform(X, y)

    def transform_tlag(self, at):
        X = self.Xh
        y = self.yh
        X, y = super().transform(X, y)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        t = len(self.slots)

        s = max(tlags)
        r = s + t

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = sx * mx + sy * my
        mu = st * my
        n = y.shape[0] - r

        Xt = np.zeros((n, mt), dtype=y.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in reversed(ylags):
                Xt[i, c:c + my] = y[t + i - j]
                c += my
            for j in reversed(xlags):
                Xt[i, c:c + mx] = X[t + i - j]
                c += mx

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[t + i + j]
                c += my
        # end

        return Xt, yt
# end


class LinearPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

    def transform(self, X: np.ndarray, fh: int):
        Xp, fh = super().transform(X, fh)

        Xh = self.Xh
        yh = self.yh

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        mx = Xh.shape[1] if Xh is not None else 0
        my = yh.shape[1]
        mt = sx * mx + sy * my
        mu = st * my

        Xt = np.zeros((1, mt), dtype=yh.dtype)
        yt = np.zeros((1, mu), dtype=yh.dtype)
        yp = np.zeros((fh, my), dtype=yh.dtype)

        self.Xt = Xt
        self.yt = yt
        self.Xp = Xp
        self.yp = yp

        return yp
    # end

    def _xat(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _yat(self, i):
        return self.yh[i, 0] if i < 0 else self.yp[i, 0]

    def step(self, i) -> np.ndarray:
        xat = self._xat
        yat = self._yat

        xlags = self.xlags if self.Xh is not None else []
        ylags = self.ylags

        mx = self.Xh.shape[1] if self.Xh is not None else 0
        my = self.yh.shape[1]
        Xt = self.Xt

        c = 0
        for j in reversed(ylags):
            Xt[0, c:c + my] = yat(i - j)
            c += my
        for j in reversed(xlags):
            Xt[0, c:c + mx] = xat(i - j)
            c += mx

        return Xt
    # end

    def update(self, i, y_pred, t=None):
        tlags = [t] if t is not None else self.tlags
        st = len(tlags)
        nfh = len(self.yp)
        my = self.yp.shape[1]

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((1, -1))

        c = 0
        for j in range(st):
            t = tlags[j]
            if i + t < nfh:
                self.yp[i + t] = y_pred[0, c:c+my]
                c += my
        # end
        return i + st

    def fit_transform(self, X, y):
        raise NotImplemented()
# end
