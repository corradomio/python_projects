import numpy as np
import pandas as pd
from pandas import DataFrame

from stdlib import NoneType
from .base import XyBaseEncoder
from .lagst import _resolve_xylags, lmax


def _to_numpy(X, dtype):
    if X is None:
        return None
    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.to_numpy(dtype=dtype)
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    if dtype is not None and X.dtype != dtype:
        X = X.astype(dtype)
    return X


# ---------------------------------------------------------------------------
# LagsArrayTransform
# ---------------------------------------------------------------------------

class LagsArrayTransform(XyBaseEncoder):

    def __init__(self,
                 xlags=None,
                 ylags=None,
                 tlags=None,
                 dtype=np.float32,
                 sequence=False,
                 channels=False):
        super().__init__(None, False)

        assert dtype is not None, "Parameter 'dtype' is None"

        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags

        self.dtype = dtype
        self.sequence = sequence
        self.channels = channels

        self._xlags = _resolve_xylags(xlags)
        self._ylags = _resolve_xylags(ylags)
        self._tlags = _resolve_xylags(tlags, True)

        self.ih = None      # Index history
        self.Xh = None      # X history
        self.yh = None      # y history
        self.Xf = None      # X forecast
        self.yf = None      # y forecast
        self.Xp = None      # X prediction
        self.yp = None      # y prediction

        self.Xp_shape = None    # X shape used in prediction
        self.yp_shape = None    # y shape used in prediction
        self.yf_shape = None    # y forecast shape

    # -----------------------------------------------------------------------

    def fit(self, X, y):
        self._check_Xy(X, y)

        X_ = _to_numpy(X, self.dtype)
        y_ = _to_numpy(y, self.dtype)

        self.ih = y.index
        self.Xh = X_
        self.yh = y_

        xlags = list(reversed(self._xlags)) if X is not None else []
        ylags = list(reversed(self._ylags)) if y is not None else []
        tlags = self._tlags

        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y
        nt = len(tlags)

        mx = X_.shape[1] if nx > 0 else 0  # n of features for x
        my = y_.shape[1] if ny > 0 else 0  # n of features for y

        if not self.sequence:
            self.Xp_shape = (1, nx*mx + ny*my)
            self.yp_shape = (1, nt*my)
        elif not self.channels:
            nf = nx if nx > 0 else ny
            self.Xp_shape = (1, nf, mx + my)
            self.yp_shape = (1, nt, my)
        else:
            nf = nx if nx > 0 else ny
            self.Xp_shape = (1, mx + my, nf)
            self.yp_shape = (1, my, nt)

        return self

    def transform(self, X: DataFrame, y: DataFrame):
        self._check_Xy(X, y)

        X_ = _to_numpy(X, self.dtype)
        y_ = _to_numpy(y, self.dtype)

        self.Xf = X_
        self.yf = y_

        if self.ih[0] == X.index[0]:
            return self._transform_fitted(X_, y_)
        elif self.ih[-1] == (X.index[0]-1):
            return self._transform_forecast(X_, y_)
        else:
            raise ValueError("Invalid timestamps")

    # -----------------------------------------------------------------------

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xf[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yf[i]

    def _transform_fitted(self, X, y):
        xlags = list(reversed(self._xlags)) if X is not None else []
        ylags = list(reversed(self._ylags)) if y is not None else []
        s = max(lmax(xlags), lmax(ylags))

        if not self.sequence:
            return self._transform_flatten(X, y, s)
        elif self.channels:
            return self._transform_channels(X, y, s)
        else:
            return self._transform_sequence(X, y, s)

    def _transform_forecast(self, X, y):
        if not self.sequence:
            # return self._transform_forecast_flatten(X, y)
            return self._transform_flatten(X, y, 0)
        elif self.channels:
            # return self._transform_forecast_channels(X, y)
            return self._transform_channels(X, y, 0)
        else:
            # return self._transform_forecast_sequence(X, y)
            return self._transform_sequence(X, y, 0)

    def _transform_flatten(self, X, y, s):
        xlags = list(reversed(self._xlags)) if X is not None else []
        ylags = list(reversed(self._ylags)) if y is not None else []
        tlags = self._tlags

        n = len(X)  # n of rows
        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y
        nt = len(tlags)  # n of lags for t

        st = max(tlags)  # window length for t)arget

        mx = X.shape[1] if nx > 0 else 0  # n of features for x
        my = y.shape[1] if ny > 0 else 0  # n of features for y
        mt = y.shape[1]
        nl = n - (st + s)  # n of predicted rows

        Xt = np.zeros((nl, nx*mx + ny*my), dtype=self.dtype)
        yt = np.zeros((nl, nt*mt), dtype=self.dtype)

        atx = self._atx
        aty = self._aty

        for j in range(n):

            c = 0
            for i in range(nx):
                k = xlags[i]
                ik = s + j - k
                Xt[j, c:c + mx] = atx(ik)
                c += mx

            for i in range(ny):
                k = ylags[i]
                ik = s + j - k
                Xt[j, c:c + my] = aty(ik)
                c += my

            c = 0
            for i in range(nt):
                k = tlags[i]
                ik = s + j + k
                yt[j, c:c + my] = aty(ik)
                c += my

        return Xt, yt

    def _transform_channels(self, X, y, s):
        Xt, yt = self._transform_sequence(X, y, s)
        Xt = Xt.swapaxes(1, 2)
        return Xt, yt

    def _transform_sequence(self, X, y, s):
        xlags = list(reversed(self._xlags)) if X is not None else []
        ylags = list(reversed(self._ylags)) if y is not None else []
        tlags = self._tlags

        n = len(X)  # n of rows
        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y
        nf = max(nx, ny)  # n of lags for both features
        nt = len(tlags)  # n of lags for t)arget

        assert nx == 0 or ny == 0 or nx == ny and xlags == ylags, "Valid x/y lags : [n, 0], [0, n], [n, n]"

        st = max(tlags)  # window length for t)arget

        mx = X.shape[1] if nx > 0 else 0  # n of features
        my = y.shape[1] if ny > 0 else 0  # n of features
        mt = y.shape[1]
        nl = n - (s + st)  # n of predicted rows

        Xt = np.zeros((nl, nf, mx + my), dtype=self.dtype)
        yt = np.zeros((nl, nt, mt), dtype=self.dtype)

        atx = self._atx
        aty = self._aty

        for j in range(n):

            for i in range(nx):
                k = xlags[i]
                ik = s + j - k
                Xt[j, i, :mx] = atx(ik)

            for i in range(ny):
                k = ylags[i]
                ik = s + j - k
                Xt[j, i, mx:] = aty(ik)

            for i in range(nt):
                k = tlags[i]
                ik = s + j + k
                yt[:, i] = aty(ik)

        return Xt, yt


# ---------------------------------------------------------------------------
# LagsArrayTransform
# ---------------------------------------------------------------------------

class LagsForecastTransform(XyBaseEncoder):

    def __init__(self,
                 xlags=None,
                 ylags=None,
                 tlags=None,
                 dtype=np.float32,
                 sequence=False,
                 channels=False):
        super().__init__(None, False)

        assert dtype is not None, "Parameter 'dtype' is None"

        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags

        self.dtype = dtype
        self.sequence = sequence
        self.channels = channels

        self._xlags = _resolve_xylags(xlags)
        self._ylags = _resolve_xylags(ylags)
        self._tlags = _resolve_xylags(tlags, True)

        self.ih = None  # Index history
        self.Xh = None  # X history
        self.yh = None  # y history
        self.Xf = None  # X forecast
        self.yf = None  # y forecast
        self.Xp = None  # X prediction
        self.yp = None  # y prediction

        self.Xp_shape = None  # X shape used in prediction
        self.yp_shape = None  # y shape used in prediction
        self.yf_shape = None  # y forecast shape

    def fit(self, *, fh=None, X=None):
        assert isinstance(fh, (NoneType, int)), "Parameter 'fh' must be int"

        X_ = _to_numpy(X, self.dtype)
        if fh is None:
            fh = len(X_)

        yh_shape = (fh,) + self.yh.shape[1:]

        self.Xf = X_
        self.yf = np.zeros(yh_shape, dtype=self.dtype)
        self.Xp = np.zeros(self.Xp_shape, dtype=self.dtype)
        self.yp = np.zeros(self.yp_shape, dtype=self.dtype)

        return self.yf

    def transform(self, i):
        if not self.sequence:
            Xp = self._transform_flatten(i)
        elif self.channels:
            Xp = self._transform_channels(i)
        else:
            Xp = self._transform_sequence(i)

        return Xp

    def _atx(self, i):
        return self.Xh[i] if i < 0 else self.Xf[i]

    def _aty(self, i):
        return self.yh[i] if i < 0 else self.yf[i]

    def _transform_flatten(self, j):
        atx = self._atx
        aty = self._aty

        xlags = list(reversed(self._xlags)) if self.Xh is not None else []
        ylags = list(reversed(self._ylags)) if self.yh is not None else []

        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y

        mx = self.Xh.shape[1] if nx > 0 else 0  # n of features
        my = self.yh.shape[1] if ny > 0 else 0  # n of features

        Xp = self.Xp

        c = 0
        for i in range(nx):
            k = xlags[i]
            ik = j - k
            Xp[0, c:c + mx] = atx(ik)
            c += mx

        for i in range(ny):
            k = ylags[i]
            ik = j - k
            Xp[0, c:c + my] = aty(ik)
            c += my

        return Xp

    def _transform_channels(self, j):
        atx = self._atx
        aty = self._aty

        xlags = list(reversed(self._xlags)) if self.Xh is not None else []
        ylags = list(reversed(self._ylags)) if self.yh is not None else []

        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y

        mx = self.Xh.shape[1] if nx > 0 else 0  # n of features

        Xp = self.Xp

        for i in range(nx):
            k = xlags[i]
            ik = j - k
            Xp[0, :mx, i] = atx(ik)

        for i in range(ny):
            k = ylags[i]
            ik = j - k
            Xp[0, mx:, i] = aty(ik)

        return Xp

    def _transform_sequence(self, j):
        atx = self._atx
        aty = self._aty

        xlags = list(reversed(self._xlags)) if self.Xh is not None else []
        ylags = list(reversed(self._ylags)) if self.yh is not None else []

        nx = len(xlags)  # n of lags for x
        ny = len(ylags)  # n of lags for y

        mx = self.Xh.shape[1] if nx > 0 else 0  # n of features

        Xp = self.Xp

        for i in range(nx):
            k = xlags[i]
            ik = j - k
            Xp[0, i, :mx] = atx(ik)

        for i in range(ny):
            k = ylags[i]
            ik = j - k
            Xp[0, i, mx:] = aty(ik)

        return Xp

    # -----------------------------------------------------------------------

    def set_forecast(self, j, y):
        if not self.sequence:
            self._set_flatten(j, y)
        else:
            self._set_sequence(j, y)

    def _set_flatten(self, j, y):
        tlags = self._tlags
        nt = len(tlags)
        my = self.yf.shape[1]
        ny = len(self.yf)

        c = 0
        for i in range(nt):
            k = tlags[i]
            ik = j + k

            # if |tlags| > 1, the prediction will be longer than yf
            if ik < ny:
                self.yf[ik] = y[0, c:c + my]
                c += my

    def _set_sequence(self, j, y):
        tlags = self.tlags
        nt = len(tlags)
        ny = len(self.yf)

        for i in range(nt):
            k = tlags[i]
            ik = j + k

            # if |tlags| > 1, the prediction will be longer than yf
            if ik < ny:
                self.yf[ik] = y[0, i]

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ArrayTransformer
# ---------------------------------------------------------------------------
# It is able to prepare the array for
#
#       linear models   2D tensor (batch, data)
#       RNN             3D tensor (batch, sequence, data)
#       CNN             3D tensor (batch, data, sequence)
#

# class ArrayTransformer(XyBaseEncoder):
#     def __init__(self,
#                  xlags=None,
#                  ylags=None,
#                  tlags=None,
#                  dtype=np.float32,
#                  sequence=False,
#                  channels=False):
#         super().__init__(None, False)
#
#         assert dtype is not None, "Parameter 'dtype' is None"
#
#         self.xlags = xlags
#         self.ylags = ylags
#         self.tlags = tlags
#
#         self.dtype = dtype
#         self.sequence = sequence
#         self.channels = channels
#
#         self._xlags = _resolve_xylags(xlags)
#         self._ylags = _resolve_xylags(ylags)
#         self._tlags = _resolve_xylags(tlags, True)
#
#         self.ih = None      # Index history
#         self.Xh = None      # X history
#         self.yh = None      # y history
#         self.Xf = None      # X forecast
#         self.yf = None      # y forecast
#         self.Xp = None      # X prediction
#         self.yp = None      # y prediction
#
#         self.Xp_shape = None    # X shape used in prediction
#         self.yp_shape = None    # y shape used in prediction
#         self.yf_shape = None    # y forecast shape
#
#     # -----------------------------------------------------------------------
#
#     def fit(self, X, y):
#         self._check_Xy(X, y)
#
#         X_ = _to_numpy(X, self.dtype)
#         y_ = _to_numpy(y, self.dtype)
#
#         self.ih = y.index
#         self.Xh = X_
#         self.yh = y_
#
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#         nt = len(tlags)
#
#         mx = X_.shape[1] if nx > 0 else 0  # n of features for x
#         my = y_.shape[1] if ny > 0 else 0  # n of features for y
#
#         if not self.sequence:
#             self.Xp_shape = (1, nx*mx + ny*my)
#             self.yp_shape = (1, nt*my)
#         elif not self.channels:
#             nf = nx if nx > 0 else ny
#             self.Xp_shape = (1, nf, mx + my)
#             self.yp_shape = (1, nt, my)
#         else:
#             nf = nx if nx > 0 else ny
#             self.Xp_shape = (1, mx + my, nf)
#             self.yp_shape = (1, my, nt)
#
#         return self
#
#     def transform(self, X: DataFrame, y: DataFrame):
#         self._check_Xy(X, y)
#
#         X_ = _to_numpy(X, self.dtype)
#         y_ = _to_numpy(y, self.dtype)
#
#         self.Xf = X_
#         self.yf = y_
#
#         if self.ih[0] == X.index[0]:
#             return self._transform_fitted(X_, y_)
#         elif self.ih[-1] == (X.index[0]-1):
#             return self._transform_forecast(X_, y_)
#         else:
#             raise ValueError("Invalid timestamps")
#
#     # -----------------------------------------------------------------------
#
#     def _transform_fitted(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         s = max(lmax(xlags), lmax(ylags))
#
#         if not self.sequence:
#             # return self._transform_fitted_flatten(X, y)
#             return self._transform_flatten(X, y, s)
#         elif self.channels:
#             # return self._transform_fitted_channels(X, y)
#             return self._transform_channels(X, y, s)
#         else:
#             # return self._transform_fitted_sequence(X, y)
#             return self._transform_sequence(X, y, s)
#
#     def _transform_fitted_flatten(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)          # n of rows
#         nx = len(xlags)     # n of lags for x
#         ny = len(ylags)     # n of lags for y
#         nt = len(tlags)     # n of lags for t
#
#         sx = lmax(xlags)    # window length for x
#         sy = lmax(ylags)    # window length for y
#         sf = max(sx, sy)    # window length for both features
#         st = max(tlags)     # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0    # n of features for x
#         my = y.shape[1] if ny > 0 else 0    # n of features for y
#         mt = y.shape[1]
#         nl = n - (sf + st)                  # n of predicted rows
#
#         Xt = np.zeros((nl, nx*mx + ny*my), dtype=self.dtype)
#         yt = np.zeros((nl, nt*mt), dtype=self.dtype)
#
#         c = 0
#         for i in range(nx):
#             k = xlags[i]
#             ik = sf - k
#             Xt[:, c:c+mx] = X[ik:ik + nl]
#             c += mx
#
#         for i in range(ny):
#             k = ylags[i]
#             ik = sf - k
#             Xt[:, c:c+my] = y[ik:ik + nl]
#             c += my
#
#         c = 0
#         for i in range(nt):
#             k = tlags[i]
#             ik = sf + k
#             yt[:, c:c+my] = y[ik:ik + nl]
#             c += my
#
#         return Xt, yt
#
#     def _transform_fitted_channels(self, X, y):
#         Xt, yt = self._transform_fitted_sequence(X, y)
#         Xt = Xt.swapaxes(1, 2)
#         return Xt, yt
#
#     def _transform_fitted_sequence(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)          # n of rows
#         nx = len(xlags)     # n of lags for x
#         ny = len(ylags)     # n of lags for y
#         nf = max(nx, ny)    # n of lags for both features
#         nt = len(tlags)     # n of lags for t)arget
#
#         assert nx == 0 or ny == 0 or nx == ny and xlags == ylags, "Valid x/y lags : [n, 0], [0, n], [n, n]"
#
#         sx = lmax(xlags)    # window length for x
#         sy = lmax(ylags)    # window length for y
#         sf = max(sx, sy)    # window length for both features
#         st = max(tlags)     # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0  # n of features
#         my = y.shape[1] if ny > 0 else 0  # n of features
#         mt = y.shape[1]
#         nl = n - (sf + st)  # n of predicted rows
#
#         Xt = np.zeros((nl, nf, mx + my), dtype=self.dtype)
#         yt = np.zeros((nl, nt, mt), dtype=self.dtype)
#
#         for i in range(nx):
#             k = xlags[i]
#             ik = sx - k
#             Xt[:, i, :mx] = X[ik:ik + nl]
#
#         for i in range(ny):
#             k = ylags[i]
#             ik = sy - k
#             Xt[:, i, mx:] = y[ik:ik + nl]
#
#         for i in range(nt):
#             k = tlags[i]
#             ik = sy + k
#             yt[:, i] = y[ik:ik + nl]
#
#         return Xt, yt
#
#     # -----------------------------------------------------------------------
#
#     def _transform_forecast(self, X, y):
#         if not self.sequence:
#             # return self._transform_forecast_flatten(X, y)
#             return self._transform_flatten(X, y, 0)
#         elif self.channels:
#             # return self._transform_forecast_channels(X, y)
#             return self._transform_channels(X, y, 0)
#         else:
#             # return self._transform_forecast_sequence(X, y)
#             return self._transform_sequence(X, y, 0)
#
#     def _transform_forecast_flatten(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#         nt = len(tlags)  # n of lags for t
#
#         st = max(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0  # n of features for x
#         my = y.shape[1] if ny > 0 else 0  # n of features for y
#         mt = y.shape[1]
#         nl = n - st  # n of predicted rows
#
#         Xt = np.zeros((nl, nx*mx + ny*my), dtype=self.dtype)
#         yt = np.zeros((nl, nt*mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(n):
#
#             c = 0
#             for i in range(nx):
#                 k = xlags[i]
#                 ik = j - k
#                 Xt[j, c:c + mx] = atx(ik)
#                 c += mx
#
#             for i in range(ny):
#                 k = ylags[i]
#                 ik = j - k
#                 Xt[j, c:c + my] = aty(ik)
#                 c += my
#
#             c = 0
#             for i in range(nt):
#                 k = tlags[i]
#                 ik = j + k
#                 yt[j, c:c + my] = aty(ik)
#                 c += my
#
#         return Xt, yt
#
#     def _transform_forecast_channels(self, X, y):
#         Xt, yt = self._transform_forecast_sequence(X, y)
#         Xt = Xt.swapaxes(1, 2)
#         return Xt, yt
#
#     def _transform_forecast_sequence(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#         nf = max(nx, ny)  # n of lags for both features
#         nt = len(tlags)  # n of lags for t)arget
#
#         assert nx == 0 or ny == 0 or nx == ny and xlags == ylags, "Valid x/y lags : [n, 0], [0, n], [n, n]"
#
#         st = max(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0  # n of features
#         my = y.shape[1] if ny > 0 else 0  # n of features
#         mt = y.shape[1]
#         nl = n - st  # n of predicted rows
#
#         Xt = np.zeros((nl, nf, mx + my), dtype=self.dtype)
#         yt = np.zeros((nl, nt, mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(n):
#
#             for i in range(nx):
#                 k = xlags[i]
#                 ik = j - k
#                 Xt[j, i, :mx] = atx(ik)
#
#             for i in range(ny):
#                 k = ylags[i]
#                 ik = j - k
#                 Xt[j, i, mx:] = aty(ik)
#
#             for i in range(nt):
#                 k = tlags[i]
#                 ik = j + k
#                 yt[:, i] = aty(ik)
#
#         return Xt, yt
#
#     # -----------------------------------------------------------------------
#
#     def _transform_flatten(self, X, y, s):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#         nt = len(tlags)  # n of lags for t
#
#         st = max(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0  # n of features for x
#         my = y.shape[1] if ny > 0 else 0  # n of features for y
#         mt = y.shape[1]
#         nl = n - (st + s)  # n of predicted rows
#
#         Xt = np.zeros((nl, nx*mx + ny*my), dtype=self.dtype)
#         yt = np.zeros((nl, nt*mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(n):
#
#             c = 0
#             for i in range(nx):
#                 k = xlags[i]
#                 ik = s + j - k
#                 Xt[j, c:c + mx] = atx(ik)
#                 c += mx
#
#             for i in range(ny):
#                 k = ylags[i]
#                 ik = s + j - k
#                 Xt[j, c:c + my] = aty(ik)
#                 c += my
#
#             c = 0
#             for i in range(nt):
#                 k = tlags[i]
#                 ik = s + j + k
#                 yt[j, c:c + my] = aty(ik)
#                 c += my
#
#         return Xt, yt
#
#     def _transform_channels(self, X, y, s):
#         Xt, yt = self._transform_sequence(X, y, s)
#         Xt = Xt.swapaxes(1, 2)
#         return Xt, yt
#
#     def _transform_sequence(self, X, y, s):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#         nf = max(nx, ny)  # n of lags for both features
#         nt = len(tlags)  # n of lags for t)arget
#
#         assert nx == 0 or ny == 0 or nx == ny and xlags == ylags, "Valid x/y lags : [n, 0], [0, n], [n, n]"
#
#         st = max(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if nx > 0 else 0  # n of features
#         my = y.shape[1] if ny > 0 else 0  # n of features
#         mt = y.shape[1]
#         nl = n - (s + st)  # n of predicted rows
#
#         Xt = np.zeros((nl, nf, mx + my), dtype=self.dtype)
#         yt = np.zeros((nl, nt, mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(n):
#
#             for i in range(nx):
#                 k = xlags[i]
#                 ik = s + j - k
#                 Xt[j, i, :mx] = atx(ik)
#
#             for i in range(ny):
#                 k = ylags[i]
#                 ik = s + j - k
#                 Xt[j, i, mx:] = aty(ik)
#
#             for i in range(nt):
#                 k = tlags[i]
#                 ik = s + j + k
#                 yt[:, i] = aty(ik)
#
#         return Xt, yt
#
#     # -----------------------------------------------------------------------
#
#     def prepare_forecast(self, *, fh=None, X=None):
#         assert isinstance(fh, (NoneType, int)), "Parameter 'fh' must be int"
#
#         X_ = _to_numpy(X, self.dtype)
#         if fh is None:
#             fh = len(X_)
#
#         yh_shape = (fh,) + self.yh.shape[1:]
#
#         self.Xf = X_
#         self.yf = np.zeros(yh_shape, dtype=self.dtype)
#         self.Xp = np.zeros(self.Xp_shape, dtype=self.dtype)
#         self.yp = np.zeros(self.yp_shape, dtype=self.dtype)
#
#         return self.yf
#
#     def forecast(self, i):
#         if not self.sequence:
#             Xp = self._prepare_forecast_flatten(i)
#         elif self.channels:
#             Xp = self._prepare_forecast_channels(i)
#         else:
#             Xp = self._prepare_forecast_sequence(i)
#
#         return Xp
#
#     def _atx(self, i):
#         return self.Xh[i] if i < 0 else self.Xf[i]
#
#     def _aty(self, i):
#         return self.yh[i] if i < 0 else self.yf[i]
#
#     def _prepare_forecast_flatten(self, j):
#         atx = self._atx
#         aty = self._aty
#
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#
#         mx = self.Xh.shape[1] if nx > 0 else 0  # n of features
#         my = self.yh.shape[1] if ny > 0 else 0  # n of features
#
#         Xp = self.Xp
#
#         c = 0
#         for i in range(nx):
#             k = xlags[i]
#             ik = j - k
#             Xp[0, c:c + mx] = atx(ik)
#             c += mx
#
#         for i in range(ny):
#             k = ylags[i]
#             ik = j - k
#             Xp[0, c:c + my] = aty(ik)
#             c += my
#
#         return Xp
#
#     def _prepare_forecast_channels(self, j):
#         atx = self._atx
#         aty = self._aty
#
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#
#         mx = self.Xh.shape[1] if nx > 0 else 0  # n of features
#
#         Xp = self.Xp
#
#         for i in range(nx):
#             k = xlags[i]
#             ik = j - k
#             Xp[0, :mx, i] = atx(ik)
#
#         for i in range(ny):
#             k = ylags[i]
#             ik = j - k
#             Xp[0, mx:, i] = aty(ik)
#
#         return Xp
#
#     def _prepare_forecast_sequence(self, j):
#         atx = self._atx
#         aty = self._aty
#
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#
#         nx = len(xlags)  # n of lags for x
#         ny = len(ylags)  # n of lags for y
#
#         mx = self.Xh.shape[1] if nx > 0 else 0  # n of features
#
#         Xp = self.Xp
#
#         for i in range(nx):
#             k = xlags[i]
#             ik = j - k
#             Xp[0, i, :mx] = atx(ik)
#
#         for i in range(ny):
#             k = ylags[i]
#             ik = j - k
#             Xp[0, i, mx:] = aty(ik)
#
#         return Xp
#
#     # -----------------------------------------------------------------------
#
#     def set_forecast(self, j, y):
#         if not self.sequence:
#             self._set_flatten(j, y)
#         else:
#             self._set_sequence(j, y)
#
#     def _set_flatten(self, j, y):
#         tlags = self._tlags
#         nt = len(tlags)
#         my = self.yf.shape[1]
#         ny = len(self.yf)
#
#         c = 0
#         for i in range(nt):
#             k = tlags[i]
#             ik = j + k
#
#             # if |tlags| > 1, the prediction will be longer than yf
#             if ik < ny:
#                 self.yf[ik] = y[0, c:c+my]
#                 c += my
#
#     def _set_sequence(self, j, y):
#         tlags = self.tlags
#         nt = len(tlags)
#         ny = len(self.yf)
#
#         for i in range(nt):
#             k = tlags[i]
#             ik = j + k
#
#             # if |tlags| > 1, the prediction will be longer than yf
#             if ik < ny:
#                 self.yf[ik] = y[0, i]
# # end
