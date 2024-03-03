import numpy as np

from deprecated import deprecated
from .base import ModelTrainTransform, ModelPredictTransform, ARRAY_OR_DF
from ..lags import lmax


# ---------------------------------------------------------------------------
# CNNTrainTransform
# CNNPredictTransform
# ---------------------------------------------------------------------------

@deprecated(reason="You should use LagsTrainTransform")
class CNNTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, xlags=None, ylags=None, tlags=(0,), flatten=False):
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)

        self.flatten = flatten
        xlags = self.xlags
        ylags = self.ylags
        assert len(xlags) == 0 or xlags == ylags, "Supported only [0, n], [n, n]"
    # end

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF = None, fh=None) -> tuple:
        X, y = self._check_Xy(X, y, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        s = len(self.slots)
        t = lmax(tlags) + 1
        r = s + t

        mx = X.shape[1] if sx > 0 else 0
        my = y.shape[1]
        mt = mx + my

        n = y.shape[0] - r

        Xt = np.zeros((n, mt, sy), dtype=y.dtype)

        for i in range(n):
            for j in range(sy):
                k = ylags[sy - 1 - j]   # reversed
                Xt[i, 0:my, j] = y[i + s - k]
            # end
        # end

        for i in range(n):
            for j in range(sx):
                k = xlags[sx - 1 - j]   # reversed
                Xt[i, my:, j] = X[i + s - k]
            # end
        # end

        yt = np.zeros((n, my, st), dtype=y.dtype)

        for i in range(n):
            # pass
            for j in range(st):
                k = tlags[j]
                yt[i, :, j] = y[i + s + k]
                # pass
            # end
        # end

        if self.flatten:
            yt = yt.reshape((yt.shape[0], -1))

        return Xt, yt
    # end
# end


@deprecated(reason="You should use LagsPredictTransform")
class CNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, xlags=None, ylags=None, tlags=(0,), flatten=False):
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)
        self.flatten = flatten
    # end

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None) -> np.ndarray:
        fh, X = super().transform(fh, X, y)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags

        sx = len(xlags)
        sy = len(ylags)

        y = self.yh

        mx = X.shape[1] if sx > 0 else 0
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
