import numpy as np
from typing import Optional

from .base import ModelTrainTransform, ModelPredictTransform
from ..lags import resolve_lags, lmax


# ---------------------------------------------------------------------------
# RNNSlotsTrainTransform
# RNNSlotsPredictTransform
# ---------------------------------------------------------------------------
# These transformers differ from RNN(Train|Predict))Transform because,
# instead to replicate 'slots.input' and 'slots.target',
# they replicate 'slots.input_list' and 'slots.target_list',
#
# The difference is, for example:
#
#       input_list = [[0], [1,2,3,4,5,6], [2,4,6,8]]
#       input = [0,1,2,3,4,5,6,8]
#
# that is, in the first case, [2,4,6] are replicated 2 times, because present
# in two sub-lists, instead in the second case, they are present a single time

class RNNSlotsTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

        #
        # override the initialization of self.xlags, self.ylags
        # because 'super' uses 'slots.input' and 'slots.target'
        #
        self.xlags = slots.input_lists
        self.ylags = slots.target_lists
    # end

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        X, y = super().transform(X, y)

        # Note: self.xlags and self.ylags ARE list of timeslots!
        #   xlags = [[0], [1,2,3,4,5], [2,4,6]]
        #   ylags = []
        xlags_list: list[list[int]] = self.xlags if X is not None else []
        ylags_list: list[list[int]] = self.ylags
        tlags = self.tlags

        t = len(self.slots)

        u = len(tlags)
        v = lmax(tlags)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        n = y.shape[0] - (t + v)

        Xts: list[np.ndarray] = []

        for xlags in xlags_list:
            s = len(xlags)

            Xt = np.zeros((n, s, mx), dtype=y.dtype)

            for i in range(n):
                for j in range(s):
                    k = xlags[s - 1 - j]
                    Xt[i, j, :] = X[i + t - k]

            Xts.append(Xt)
        # end

        for ylags in ylags_list:
            s = len(ylags)

            Xt = np.zeros((n, s, my), dtype=y.dtype)

            for i in range(n):
                for j in range(s):
                    k = ylags[s - 1 - j]
                    Xt[i, j, :] = y[i + t - k]
                # end
            # end

            Xts.append(Xt)
        # end

        yt = np.zeros((n, my), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in range(u):
                k = tlags[j]
                yt[i, c:c + my] = y[i + t + k]
                c += my
            # end
        # end

        return Xts, yt
    # end

    # def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    #     return self.fit(X, y).transform(X, y)
# end


class RNNSlotsPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

        #
        # override the initialization of self.xlags, self.ylags
        # because 'super' uses 'slots.input' and 'slots.target'
        #
        self.xlags = slots.input_lists
        self.ylags = slots.target_lists

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        super().transform(X, fh)

        xlags_list = self.xlags if X is not None else []
        ylags_list = self.ylags
        y = self.yh
        self.Xp = X

        if fh == 0: fh = len(X)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]

        Xts = []
        for xlags in xlags_list:
            s = len(xlags)
            Xt = np.zeros((1, s, mx), dtype=y.dtype)
            Xts.append(Xt)

        for ylags in ylags_list:
            s = len(ylags)
            Xt = np.zeros((1, s, my), dtype=y.dtype)
            Xts.append(Xt)

        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xt = Xts
        self.yp = yp

        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i, 0]

    def step(self, i) -> list[np.ndarray]:
        atx = self._atx
        aty = self._aty

        Xts = self.Xt

        sx = len(self.xlags if self.Xh is not None else [])
        sy = len(self.ylags)

        for l in range(sx):
            Xt = Xts[l]
            xlags = self.xlags[l]
            s = len(xlags)

            for j in range(s):
                k = xlags[s - 1 - j]
                Xt[0, j, :] = atx(i - k)

        for l in range(sy):
            Xt = Xts[sx+l]
            ylags = self.ylags[l]
            s = len(ylags)

            for j in range(s):
                k = ylags[s - 1 - j]
                Xt[0, j, :] = aty(i - k)

        return Xts
    # end

    def update(self, i, y_pred):
        self.yp[i] = y_pred[0]
        return i+1
# end
