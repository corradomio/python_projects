from typing import Optional

import numpy as np
from .model_transform import ModelTrainTransform, ModelPredictTransform


class RNNTrainTransform3D(ModelTrainTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

        #
        # check if the lags are in the correct form:
        #
        #   (x|y)lags = [1*d, 2*d, 3*d, ...]
        #
        xlags = self.xlags
        ylags = self.ylags
        d = ylags[0]
        llags = list(range(d, (len(ylags)+1)*d, d))

        assert len(xlags) == 0 or xlags == ylags, "Supported only [0, n], [n, n]"
        assert ylags == llags, "Supported only [d, 2*d, 3*d, ...]"
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
                k = ylags[sx - 1 - j]  # reversed

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

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class RNNPredictTransform3D(ModelPredictTransform):

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
        self.Xp = X

        if fh == 0: fh = len(X)

        sx = len(xlags)
        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my

        Xt = np.zeros((1, sy, mt), dtype=y.dtype)
        yp = np.zeros((fh, st, my), dtype=y.dtype)

        self.Xt = Xt
        self.yp = yp

        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i, 0]

    def step(self, i) -> np.ndarray:
        atx = self._atx
        aty = self._aty

        X = self.Xh
        y = self.yh
        Xt = self.Xt

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        my = y.shape[1]

        sy = len(ylags)
        sx = len(xlags)

        for j in range(sy):
            k = ylags[sy - 1 - j]    # reversed
            Xt[0, j, 0:my] = aty(i - k)
        # end

        for j in range(sx):
            k = ylags[sx - 1 - j]  # reversed
            Xt[0, j, my:] = atx(i - k)
        # end

        return Xt
    # end
# end
