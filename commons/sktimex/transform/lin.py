import numpy as np

from deprecated import deprecated
from .base import ModelTrainTransform, ModelPredictTransform, ARRAY_OR_DF
from ..lags import lmax


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

@deprecated(reason="You should use LagsTrainTransform")
class LinearTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, xlags=None, ylags=None, tlags=(0,)):
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)
    # end

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF=None, fh=None) -> tuple:
        X, y = self._check_Xy(X, y, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        s = len(self.slots)
        t = lmax(tlags) + 1
        r = t + s

        mx = X.shape[1] if sx > 0 else 0
        my = y.shape[1]
        mt = sx * mx + sy * my
        mu = st * my
        n = y.shape[0] - r

        Xt = np.zeros((n, mt), dtype=y.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in reversed(ylags):
                Xt[i, c:c + my] = y[s + i - j]
                c += my
            for j in reversed(xlags):
                Xt[i, c:c + mx] = X[s + i - j]
                c += mx

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[s + i + j]
                c += my
        # end

        return Xt, yt
    # end
# end


@deprecated(reason="You should use LagsPredictTransform")
class LinearPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, xlags=None, ylags=None, tlags=(0,)):
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)
    # end

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None):
        fh, X = super().transform(fh, X, y)

        Xh = self.Xh
        yh = self.yh

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        mx = Xh.shape[1] if sx > 0 else 0
        my = yh.shape[1]
        mt = sx * mx + sy * my
        mu = st * my

        Xt = np.zeros((1, mt), dtype=yh.dtype)
        yt = np.zeros((1, mu), dtype=yh.dtype)
        yp = np.zeros((fh, my), dtype=yh.dtype)

        self.Xt = Xt
        self.yt = yt
        self.Xp = X
        self.yp = yp

        return self.to_pandas(yp)
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

    def update(self, i, y_pred, t=None):
        return super().update(i, y_pred, t)
# end