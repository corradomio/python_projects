from typing import Optional

import numpy as np
from stdlib import NoneType


# ---------------------------------------------------------------------------
# LinearTrainTransform
# LinearPredictTransform
# ---------------------------------------------------------------------------

def lags_max(xlags, ylags):
    return max(
        max(xlags) if len(xlags) > 0 else 0,
        max(ylags) if len(ylags) > 0 else 0
    )


class ModelTransform:
    pass


class ModelTrainTransform(ModelTransform):

    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1], tlags=[0]):
        assert isinstance(steps, int)
        assert isinstance(xlags, list)
        assert isinstance(ylags, list)
        assert isinstance(tlags, list)
        pass


class ModePredictTransform(ModelTransform):

    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1], tlags=[0]):
        assert isinstance(steps, int)
        assert isinstance(xlags, list)
        assert isinstance(ylags, list)
        assert isinstance(tlags, list)
        pass


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
    def __init__(self, xlags: list[int] = [0], ylags: list[int] = [], tlags=[0]):
        super().__init__(1, xlags, ylags, tlags)

        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags
        self.t = lags_max(xlags, ylags)

    def __len__(self):
        return self.t

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []

        return self

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        xlags = self.xlags
        ylags = self.ylags
        tlags = self.tlags

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        s = lags_max(xlags, ylags)
        t = max(tlags)
        r = s + t

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = len(xlags) * mx + len(ylags) * my
        mu = len(tlags) * my
        n = y.shape[0] - r

        Xt = np.zeros((n, mt), dtype=y.dtype)
        yt = np.zeros((n, mu), dtype=y.dtype)

        for i in range(n):
            c = 0
            for j in ylags:
                Xt[i, c:c + my] = y[s + i - j]
                c += my
            for j in xlags:
                Xt[i, c:c + mx] = X[s + i - j]
                c += mx

            c = 0
            for j in tlags:
                yt[i, c:c + my] = y[s + i + j]
                c += my
        # end

        return Xt, yt
    # end
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
# end


class LinearPredictTransform(ModePredictTransform):

    def __init__(self, xlags: list[int] = [0], ylags: list[int] = [], tlags=[0]):
        super().__init__(1, xlags, ylags, tlags)

        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags
        self.t = lags_max(xlags, ylags)

        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None

    def __len__(self):
        return self.t

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []
        elif len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return self

    def transform(self, X: np.ndarray, fh: int) -> np.ndarray:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(fh, int)

        Xh = self.Xh
        yh = self.yh

        xlags = self.xlags
        ylags = self.ylags
        tlags = self.tlags

        s = lags_max(xlags, ylags)
        t = max(tlags)

        mx = Xh.shape[1] if Xh is not None else 0
        my = yh.shape[1]
        mt = len(xlags) * mx + len(ylags) * my
        mu = len(tlags) * my

        yp = np.zeros((fh, mu), dtype=yh.dtype)
        Xt = np.zeros((1, mt), dtype=Xh.dtype)

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
        xlags = self.xlags
        ylags = self.ylags
        mx = self.Xh.shape[1]
        my = self.yh.shape[1]
        Xt = self.Xt

        c = 0
        for j in ylags:
            Xt[0, c:c + my] = yat(i - j)
            c += my
        for j in xlags:
            Xt[0, c:c + mx] = xat(i - j)
            c += mx

        return Xt
    # end

    def fit_transform(self, X, y):
        raise NotImplemented()
# end


# ---------------------------------------------------------------------------
# RNNTrainTransform
# RNNFlatTrainTransform
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

class RNNTrainTransform(ModelTrainTransform):
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []

        return self

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = t + (s - 1)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)
        n = y.shape[0] - r

        Xt = np.zeros((n, s, mt), dtype=y.dtype)
        yt = np.zeros((n, s, my), dtype=y.dtype)

        for i in range(n):
            for j in range(s):
                c = 0
                for k in ylags:
                    Xt[i, j, c:c + my] = y[i + j + t - k]
                    c += my
                for k in xlags:
                    Xt[i, j, c:c + mx] = X[i + j + t - k]
                    c += mx

                yt[i, j] = y[i + j + t]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class RNNFlatTrainTransform(ModelTrainTransform):
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []

        return self

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = t + (s - 1)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)
        n = y.shape[0] - r

        Xt = np.zeros((n, s, mt), dtype=X.dtype)
        yt = np.zeros((n, my), dtype=y.dtype)

        for i in range(n):
            for j in range(s):
                c = 0
                for k in ylags:
                    Xt[i, j, c:c + my] = y[i + j + t - k]
                    c += my
                for k in xlags:
                    Xt[i, j, c:c + mx] = X[i + j + t - k]
                    c += mx

            yt[i] = y[i + r]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class RNNPredictTransform(ModePredictTransform):

    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []
        elif len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return self
    # end

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        xlags = self.xlags
        ylags = self.ylags
        y = self.yh

        if fh == 0: fh = len(X)

        self.Xp = X

        s = self.steps

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)

        Xt = np.zeros((1, s, mt), dtype=X.dtype)
        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xt = Xt
        self.yp = yp
        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i]

    def step(self, i):
        atx = self._atx
        aty = self._aty

        X = self.Xh
        y = self.yh
        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]

        Xt = self.Xt

        for j in range(s):
            c = 0
            for k in ylags:
                Xt[0, j, c:c + my] = aty(i + j - k - s + 1)
                c += my
            for k in xlags:
                Xt[0, j, c:c + mx] = atx(i + j - k - s + 1)
                c += mx
        # end

        return Xt
    # end
# end


# ---------------------------------------------------------------------------
# CNNTrainTransform
# CNNFlatTrainTransform
# CNNPredictTransform
# ---------------------------------------------------------------------------
# N, Channels, Channel_Length

class CNNTrainTransform(ModelTrainTransform):
    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []

        return self

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = t + (s - 1)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)
        n = y.shape[0] - r

        Xt = np.zeros((n, mt, s), dtype=X.dtype)
        yt = np.zeros((n, my, s), dtype=y.dtype)

        for i in range(n):
            c = 0
            for k in ylags:
                for j in range(s):
                    Xt[i, c:c + my, j] = y[i + j + t - k]
                c += my
            for k in xlags:
                for j in range(s):
                    Xt[i, c:c + mx, j] = X[i + j + t - k]
                c += mx

            for j in range(s):
                yt[i, :, j] = y[i + j + t]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class CNNFlatTrainTransform(ModelTrainTransform):

    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

    def __len__(self):
        return self.steps + 1

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []

        return self

    def transform(self, X: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is not None and len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        xlags = self.xlags
        ylags = self.ylags

        s = self.steps
        t = self.t
        r = t + (s - 1)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)
        n = y.shape[0] - r

        Xt = np.zeros((n, mt, s), dtype=X.dtype)
        yt = np.zeros((n, my), dtype=y.dtype)

        for i in range(n):
            c = 0
            for k in ylags:
                for j in range(s):
                    Xt[i, c:c + my, j] = y[i + j + t - k]
                c += my
            for k in xlags:
                for j in range(s):
                    Xt[i, c:c + mx, j] = X[i + j + t - k]
                c += mx

            yt[i] = y[i + r]
        # end

        return Xt, yt
    # end

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)
# end


class CNNPredictTransform(ModePredictTransform):

    def __init__(self, steps: int = 1, xlags: list[int] = [1], ylags: list[int] = [1]):
        super().__init__(steps, xlags, ylags)

        self.steps = steps
        self.xlags = xlags
        self.ylags = ylags
        self.t = lags_max(xlags, ylags)

        self.Xh = None
        self.yh = None
        self.Xp = None
        self.yp = None
        self.Xt = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, np.ndarray)

        if X is None:
            self.xlags = []
        elif len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return self

    def transform(self, X: np.ndarray, fh: int = 0) -> np.ndarray:
        assert X is None and fh > 0 or X is not None and fh == 0 or len(X) == fh

        xlags = self.xlags
        ylags = self.ylags
        y = self.yh

        if fh == 0: fh = len(X)

        self.Xp = X

        s = self.steps

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        mt = mx * len(xlags) + my * len(ylags)

        Xt = np.zeros((1, mt, s), dtype=X.dtype)
        yp = np.zeros((fh, my), dtype=y.dtype)

        self.Xt = Xt
        self.yp = yp
        return yp
    # end

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yp[i]

    def step(self, i):
        atx = self._atx
        aty = self._aty

        X = self.Xh
        y = self.yh
        xlags = self.xlags
        ylags = self.ylags

        s = self.steps

        mx = X.shape[1]
        my = y.shape[1]

        Xt = self.Xt

        c = 0
        for k in ylags:
            for j in range(s):
                Xt[0, c:c + my, j] = aty(i + j - k - s + 1)
            c += my
        for k in xlags:
            for j in range(s):
                Xt[0, c:c + mx, j] = atx(i + j - k - s + 1)
            c += mx

        return Xt
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
