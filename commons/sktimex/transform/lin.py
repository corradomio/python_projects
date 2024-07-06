import numpy as np

from ._base import TimeseriesTransform, ARRAY_OR_DF
from ._lags import lmax, tlags_start


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

class LinearTrainTransform(TimeseriesTransform):

    def __init__(self, xlags=None, ylags=None, tlags=(0,), flatten=True):
        self.xlags: list = xlags
        self.ylags: list = ylags
        self.tlags: list = tlags

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF=None, fh=None) -> tuple:
        X, y = self._check_Xy(X, y, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        s = max(lmax(xlags), lmax(ylags))
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


class LinearPredictTransform(TimeseriesTransform):

    def __init__(self, xlags=None, ylags=None, tlags=(0,), flatten=True):
        self.xlags: list = xlags
        self.ylags: list = ylags
        self.tlags: list = tlags
        self.tstart: int = tlags_start(tlags)

        self.Xh = None  # X history
        self.yh = None  # y history

        self.Xp = None  # X prediction past
        self.yp = None  # y prediction future
    # end

    def fit(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None):
        self.Xh = X
        self.yh = y
        return self
    # end

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None):
        X, y = self._check_Xy(X, y)

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

    def update(self, i, y_pred, t=None):
        #
        # Note:
        #   the parameter 't' is used to override tlags
        #   'tlags' is at minimum [0]
        #
        # Extension:
        #   it is possible to use tlags=[-3,-2,-1,0,1]
        #   in this case, it is necessary to start with the position '3'
        #   and advance 'i' ONLY of 2 slots.
        #   Really usable slots: [0,1]
        assert isinstance(i, (int, np.int32)), "The argument 'i' must be the location update (an integer)"

        tlags = self.tlags if t is None else [t]
        tstart = self.tstart if t is None else 0

        st = len(tlags)         # length of tlags
        mt = max(tlags)         # max tlags index
        nfh = len(self.yp)      # length of fh

        for j in range(tstart, st):
            k = i + tlags[j]
            if k < nfh:
                try:
                    if y_pred.ndim == 1:
                        self.yp[k] = y_pred[0]
                    else:
                        self.yp[k] = y_pred[0, j]
                except IndexError:
                    pass

        return i + mt + 1
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
