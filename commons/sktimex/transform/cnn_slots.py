import numpy as np
from typing import Optional

from .base import ModelTrainTransform, ModelPredictTransform
from ..lag import resolve_lags, flatten_max


# ---------------------------------------------------------------------------
# CNNSlotsTrainTransform
# CNNSlotsPredictTransform
# ---------------------------------------------------------------------------

class CNNSlotsTrainTransform(ModelTrainTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

        #
        # force the initialization of self.xlags, self.ylags
        # because 'super' initialization uses 'slots.input' and 'slots.target'
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
        r = max(flatten_max(xlags_list), flatten_max(ylags_list))
        t = r

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        n = y.shape[0] - r - 1

        Xts: list[np.ndarray] = []
        for xlags in xlags_list:
            s = len(xlags)

            Xt = np.zeros((n, mx, s), dtype=y.dtype)

            for i in range(n):
                for j in range(s):
                    k = xlags[s - 1 - j]
                    Xt[i, :, j] = X[i + t - k]

            Xts.append(Xt)
        # end

        for ylags in ylags_list:
            s = len(ylags)

            Xt = np.zeros((n, my, s), dtype=y.dtype)

            for i in range(n):
                for j in range(s):
                    k = ylags[s - 1 - j]
                    Xt[i, :, j] = y[i + t - k]

            Xts.append(Xt)
        # end

        yt = np.zeros((n, my), dtype=y.dtype)

        for i in range(n):
            yt[i] = y[i + r]
        # end

        return Xts, yt

    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        return self.fit(X, y).transform(X, y)


# end


class CNNSlotsPredictTransform(ModelPredictTransform):

    def __init__(self, slots=None, tlags=(0,), lags=None):
        super().__init__(
            slots=slots if slots is not None else resolve_lags(lags),
            tlags=tlags)

        #
        # force the initialization of self.xlags, self.ylags
        # because 'super' initialization uses 'slots.input' and 'slots.target'
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
            Xt = np.zeros((1, mx, s), dtype=y.dtype)
            Xts.append(Xt)

        for ylags in ylags_list:
            s = len(ylags)
            Xt = np.zeros((1, my, s), dtype=y.dtype)
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

        nxl = len(self.xlags if Xts is not None else [])
        nyl = len(self.ylags)

        for l in range(nxl):
            Xt = Xts[l]
            xlags = self.xlags[l]
            s = len(xlags)

            for j in range(s):
                k = xlags[s - 1 - j]
                Xt[0, :, j] = atx(i - k)

        for l in range(nyl):
            Xt = Xts[nxl + l]
            ylags = self.ylags[l]
            s = len(ylags)

            for j in range(s):
                k = ylags[s - 1 - j]
                Xt[0, :, j] = aty(i - k)

        return Xts
    # end
# end