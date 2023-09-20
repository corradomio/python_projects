from typing import Optional

import numpy as np

from .lag import LagSlots, flatten_max
from stdlib import NoneType


# ---------------------------------------------------------------------------
#   ModelTransform
#       ModelTrainTransform
#       ModePredictTransform
# ---------------------------------------------------------------------------

class ModelTransform:
    pass


class ModelTrainTransform(ModelTransform):

    def __init__(self, slots, tlags=(0,)):
        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list))

        self.slots = slots
        self.xlags: list = slots.input
        self.ylags: list = slots.target
        self.tlags: list = list(tlags)
    # end

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        return self
    # end

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        return X, y
    # end
# end


class ModelPredictTransform(ModelTransform):

    def __init__(self, slots, tlags=(0,)):
        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list))

        self.slots = slots
        self.xlags = slots.input
        self.ylags = slots.target
        self.tlags = tlags

        self.Xh = None  # X history
        self.yh = None  # y history
        self.Xp = None  # X prediction
        self.yp = None  # y prediction
        self.Xt = None  # X transform
        self.yt = None  # y transform
    # end

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            pass
        elif len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return self
    # end

    def transform(self, X: np.ndarray, fh: int):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(fh, int)
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if X is not None and fh == 0:
            fh = len(X)

        return X, fh
    # end
# end


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

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
# end


class LinearPredictTransform(ModelPredictTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        X, fh = super().transform(X, fh)

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
        yp = np.zeros((fh, mu), dtype=yh.dtype)

        self.Xp = X
        self.yp = yp
        self.Xt = Xt

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

    def fit_transform(self, X, y):
        raise NotImplemented()
# end


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

        assert len(xlags) == 0 or xlags == ylags
        assert ylags == llags
    # end

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().transform(X, y)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        s = len(ylags)
        t = max(ylags)

        v = max(tlags)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        mu = my * len(tlags)
        n = y.shape[0] - (t + v)

        Xt = np.zeros((n, mt, s), dtype=y.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            for j in range(s):
                k = ylags[s - 1 - j]  # reversed

                Xt[i, 0:my, j] = y[i + t - k]

                if X is not None and sx > 0:
                    Xt[i, my:, j] = X[i + t - k]
            # end

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[t + i + j]
                c += my
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class CNNPredictTransform(ModelPredictTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        X, fh = super().transform(X, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        s = len(ylags)

        y = self.yh

        if fh == 0: fh = len(X)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx + my
        mu = my * len(tlags)

        Xt = np.zeros((1, mt, s), dtype=y.dtype)
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

        X = self.Xh
        y = self.yh

        xlags = self.xlags
        ylags = self.ylags
        
        my = y.shape[1]
        s = len(ylags)
        sx = len(xlags)

        Xt = self.Xt

        for j in range(s):
            k = ylags[s - 1 - j]  # reversed

            Xt[0, 0:my, j] = aty(i - k)

            if X is not None and sx > 0:
                Xt[0, my:, j] = atx(i - k)
        # end

        return Xt
    # end
# end


# ---------------------------------------------------------------------------
# CNNSlotsTrainTransform
# CNNSlotsPredictTransform
# ---------------------------------------------------------------------------

class CNNSlotsTrainTransform(ModelTrainTransform):

    def __init__(self, slots=LagSlots(), tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

        #
        # force the initialization of self.xlags, self.ylags
        # because 'super' initialization uses 'slots.input' and 'slots.target'
        #
        self.xlags = slots.input_lists
        self.ylags = slots.target_lists

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

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

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
            Xt = Xts[nxl+l]
            ylags = self.ylags[l]
            s = len(ylags)

            for j in range(s):
                k = ylags[s - 1 - j]
                Xt[0, :, j] = aty(i - k)

        return Xts
    # end
# end


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
                Xt[i, j, my:] = y[i + t - k]
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
        self.Xp = X

        if fh == 0: fh = len(X)

        mx = X.shape[1] if X is not None and sx > 0 else 0
        my = y.shape[1]
        mt = mx + my
        mu = my * st

        Xt = np.zeros((1, sy, mt), dtype=y.dtype)
        yp = np.zeros((fh, mu), dtype=y.dtype)

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

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

        #
        # override the initialization of self.xlags, self.ylags
        # because 'super' uses 'slots.input' and 'slots.target'
        #
        self.xlags = slots.input_lists
        self.ylags = slots.target_lists

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
        v = max(tlags)

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

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class RNNSlotsPredictTransform(ModelPredictTransform):

    def __init__(self, slots, tlags=(0,)):
        super().__init__(slots=slots, tlags=tlags)

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
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
