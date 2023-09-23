from .base import *


# ---------------------------------------------------------------------------
# RNNTrainTransform
# RNNPredictTransform
# ---------------------------------------------------------------------------
# N, Sequence_Length, Data
#
# X[0]      -> (X[0],X[1],X[2],...)
# X[1]      -> (X[1],X[2],X[3],...)
# X[2]      -> (X[2],X[3],X[4],...)
#
#   y[-1]            -> y[0]
#   X[-1]            -> y[0]
#   X[-1],y[-1]      -> y[0]
#   X[-1],y[-1],X[0] -> y[0]
#
# xlags: [], [1], [0], [0,1]
# ylags: [], [1]
#
#
# Difference between RNNTrainTransform AND RNNFlatTrainTransform
# --------------------------------------------------------------
# The only difference is in y:  y[n, s, my]  vs  y[n, my]
# and how y is initialized
#
# Lags supported
# --------------
# RNN & CNN, to be consistent with the NN logic, support ONLY lags of the following form
#
#       [1,2,3,4,...]
#       [2,4,6,8,...]
#
# that is, only sequences with constant step

class RNNTrainTransform(ModelTrainTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

        #
        # check if the lags are in the correct form:
        #
        #   (x|y)lags = [1*d, 2*d, 3*d, ...]
        #
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
        v = max(tlags)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        mu = my * st

        n = y.shape[0] - (t + v)

        Xt = np.zeros((n, sy, mt), dtype=y.dtype)

        for i in range(n):
            for j in range(sy):
                k = ylags[sy - 1 - j]   # reversed
                Xt[i, j, 0:my] = y[i + t - k]
            # end
        # end

        for i in range(n):
            for j in range(sx):
                k = xlags[sx - 1 - j]  # reversed
                Xt[i, j, my:] = X[i + t - k]
            # end
        # end

        yt = np.zeros((n, mu), dtype=y.dtype)

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


class RNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

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

        Xt = np.zeros((1, sy, mt), dtype=y.dtype)
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
            Xt[0, j, 0:my] = aty(i - k)
        # end

        for j in range(sx):
            k = xlags[sx - 1 - j]   # reversed
            Xt[0, j, my:] = atx(i - k)
        # end

        return Xt
    # end
# end

