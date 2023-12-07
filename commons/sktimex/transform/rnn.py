from typing import Optional

import numpy as np

from .base import ModelTrainTransform, ModelPredictTransform, ARRAY_OR_DF
from ..lags import lmax


# ---------------------------------------------------------------------------
# RNNTrainTransform
# RNNPredictTransform
# ---------------------------------------------------------------------------

class RNNTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None, xlags=None, ylags=None, flatten=False, ytrain=False):
        """

        :param slots:
        :param tlags:
        :param lags:    alternative to slots
        :param xlags:   alternative to slots (with ylags)
        :param ylags:   alternative to slots (with xlags)
        :param flatten: it to return yt as 2D tensor
        :param ytrain:  if to return y used in train
        """
        if ylags is not None:
            slots = [xlags, ylags]
        elif lags is not None:
            slots = lags
        super().__init__(
            slots=slots,
            tlags=tlags)

        self.flatten = flatten
        self.ytrain = ytrain
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

        t = len(self.slots)
        v = lmax(tlags)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my

        n = y.shape[0] - (t + v)

        Xt = np.zeros((n, sy, mt), dtype=y.dtype)

        for i in range(n):
            for j in range(sy):
                k = ylags[sy - 1 - j]  # reversed
                Xt[i, j, 0:my] = y[i + t - k]
            # end
        # end

        for i in range(n):
            for j in range(sx):
                k = xlags[sx - 1 - j]  # reversed
                Xt[i, j, my:] = X[i + t - k]
            # end
        # end

        yt = np.zeros((n, st, my), dtype=y.dtype)

        for i in range(n):
            for j in range(st):
                k = tlags[j]
                yt[i, j, :] = y[i + t + k]
            # end
        # end

        if self.flatten:
            yt = yt.reshape((yt.shape[0], -1))

        if self.ytrain:
            yx = Xt[:, :, :my]
            return Xt, (yx, yt)
        else:
            return Xt, yt
    # end
# end


class RNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None, xlags=None, ylags=None, flatten=False):
        if ylags is not None:
            slots = [xlags, ylags]
        elif lags is not None:
            slots = lags
        super().__init__(
            slots=slots,
            tlags=tlags)

        self.flatten = flatten

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None) -> np.ndarray:
        fh, X = super().transform(fh, X, y)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags

        sx = len(xlags)
        sy = len(ylags)

        y = self.yh

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        #

        Xt = np.zeros((1, sy, mt), dtype=y.dtype)
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
            Xt[0, j, 0:my] = aty(i - k)
        # end

        for j in range(sx):
            k = xlags[sx - 1 - j]   # reversed
            Xt[0, j, my:] = atx(i - k)
        # end

        return Xt

    def update(self, i, y_pred, t=None):
        return super().update(i, y_pred, t)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
