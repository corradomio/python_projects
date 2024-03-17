# import numpy as np
# import pandas as pd
#
# from .lagst import _resolve_xylags, lmax
# from .base import XyBaseEncoder
#
#
# def _to_numpy(X, dtype):
#     if X is None:
#         return None
#     if isinstance(X, (pd.Series, pd.DataFrame)):
#         X = X.to_numpy(dtype=dtype)
#     if len(X.shape) == 1:
#         X = X.reshape((-1, 1))
#     if dtype is not None and X.dtype != dtype:
#         X = X.astype(dtype)
#     return X
#
#
# # ---------------------------------------------------------------------------
# # LagsArrayForecaster
# # ---------------------------------------------------------------------------
#
# class LagsArrayForecaster(XyBaseEncoder):
#
#     def __init__(self,
#                  xlags=None,
#                  ylags=None,
#                  tlags=None,
#                  dtype=np.float32,
#                  temporal=False,
#                  channels=False,
#                  y_flatten=False):
#         super().__init__(None, False)
#
#         assert dtype is not None, "Parameter 'dtype' must be not None"
#         assert (not temporal and not channels
#                 or temporal and not channels
#                 or not temporal and channels), \
#             "Only a parameter between 'temporal' and 'channels' can be True"
#
#         self.xlags = xlags
#         self.ylags = ylags
#         self.tlags = tlags
#
#         self.dtype = dtype
#         self.temporal = temporal
#         self.channels = channels
#         self.y_flatten = y_flatten
#
#         self._xlags = _resolve_xylags(xlags)
#         self._ylags = _resolve_xylags(ylags)
#         self._tlags = _resolve_xylags(tlags, True)
#
#         self.X = None
#         self.y = None
#         self.Xh = None  # X history
#         self.yh = None  # y history
#         self.Xf = None  # X forecast
#         self.yf = None  # y forecast
#         self.Xp = None  # X prediction
#
#         self.Xp_shape = None  # X shape used in prediction
#         self.yp_shape = None  # y shape used in prediction
#         self.yf_shape = None  # y forecast shape
#
#     def fit(self, X, y=None):
#         if isinstance(X, pd.PeriodIndex):
#             fh = X
#             X = pd.DataFrame(index=fh)
#
#         X, y = self._check_Xy(X, y)
#
#         self.X = X
#         self.y = y
#
#         X_ = _to_numpy(X, self.dtype)
#         y_ = _to_numpy(y, self.dtype)
#
#         self.Xh = X_
#         self.yh = y_
#
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#         st = len(tlags)
#
#         mx = X_.shape[1] if sx > 0 else 0  # n of features for x
#         my = y_.shape[1] if sy > 0 else 0  # n of features for y
#
#         if self.temporal:
#             nf = sx if sx > 0 else sy
#             self.Xp_shape = (1, nf, mx + my)
#             self.yp_shape = (1, st, my)
#         elif self.channels:
#             nf = sx if sx > 0 else sy
#             self.Xp_shape = (1, mx + my, nf)
#             self.yp_shape = (1, my, st)
#         else:
#             self.Xp_shape = (1, sx * mx + sy * my)
#             self.yp_shape = (1, st * my)
#
#         return self
#
#     def transform(self, X, y=None):
#         """
#         Initialize the forecaster with the input X
#         It returns a shared array that will contain the predicted values and used
#         to create the X tensor passed to the model. The array is updated by 'update(i, y)'
#
#         :param X: dataframe to use as input
#         :return: the array used as 'y' predicted.
#         """
#         if isinstance(X, pd.PeriodIndex):
#             ix = X
#             X = pd.DataFrame(index=ix)
#
#         X = self._check_X(X)
#
#         X_ = _to_numpy(X, self.dtype)
#
#         fh = len(X)
#         yh_shape = (fh,) + self.yh.shape[1:]
#
#         self.Xf = X_
#         self.yf = np.zeros(yh_shape, dtype=self.dtype)
#         self.Xp = np.zeros(self.Xp_shape, dtype=self.dtype)
#
#         return pd.DataFrame(self.yf, columns=self.y.columns, index=X.index)
#
#     def fit_transform(self, X, y=None):
#         raise NotImplemented("fit_transform is not supported")
#
#     # -----------------------------------------------------------------------
#
#     def _atx(self, i):
#         return self.Xh[i] if i < 0 else self.Xf[i]
#
#     def _aty(self, i):
#         return self.yh[i] if i < 0 else self.yf[i]
#
#     def _transform_flatten(self, j):
#         atx = self._atx
#         aty = self._aty
#
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#
#         mx = self.Xh.shape[1] if sx > 0 else 0  # n of features
#         my = self.yh.shape[1] if sy > 0 else 0  # n of features
#
#         Xp = self.Xp
#
#         c = 0
#         for i in range(sx):
#             k = xlags[i]
#             ik = j - k
#             Xp[0, c:c + mx] = atx(ik)
#             c += mx
#
#         for i in range(sy):
#             k = ylags[i]
#             ik = j - k
#             Xp[0, c:c + my] = aty(ik)
#             c += my
#
#         return Xp
#
#     def _transform_channels(self, j):
#         atx = self._atx
#         aty = self._aty
#
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#
#         mx = self.Xh.shape[1] if sx > 0 else 0  # n of features
#
#         Xp = self.Xp
#
#         for i in range(sx):
#             k = xlags[i]
#             ik = j - k
#             Xp[0, :mx, i] = atx(ik)
#
#         for i in range(sy):
#             k = ylags[i]
#             ik = j - k
#             Xp[0, mx:, i] = aty(ik)
#
#         return Xp
#
#     def _transform_temporal(self, j):
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
#     def step(self, i):
#         """
#         Prepare the tensor to use as input for the NN model.
#
#         Note: the tensor returned is a local object override at each step
#
#         :param i: current step
#         :return: the tensor to use
#         """
#         assert isinstance(i, int)
#         if self.temporal:
#             Xp = self._transform_temporal(i)
#         elif self.channels:
#             Xp = self._transform_channels(i)
#         else:
#             Xp = self._transform_flatten(i)
#
#         return Xp
#
#     def update(self, i, y) -> int:
#         """
#         Update the shared 'yf' created and returned with 'transform(X)'.
#         Return the next valid step index to use.
#
#         It updates 'yf[i+k]' with k specified in 'tlags'
#
#         Note: the number of steps to advance depends on 'tlags':
#
#             [0]         1 step
#             [0,1,2,3]   4 steps
#             [0,1,4,5]   2 steps because it starts with 0 and the longest consecutive sequence is [0,1]
#             [1,2,3]     0 steps because it doesn't start with 0
#
#         Note: if tlags is not consecutive, the values not in sequence will be override
#
#         :param i: current step
#         :param y:
#         :return: the next valid index to use with 'step(i)'
#         """
#         if len(y.shape) == 2:
#             next_i = self._update_flatten(i, y)
#         elif self.temporal:
#             next_i = self._update_temporal(i, y)
#         elif self.channels:
#             next_i = self._update_channels(i, y)
#         else:
#             next_i = self._update_flatten(i, y)
#
#         return i + self._advance()
#
#     def _update_flatten(self, i, y):
#         tlags = self._tlags
#         yf = self.yf
#
#         st = len(tlags)
#         my = yf.shape[1]
#         ny = len(yf)
#
#         c = 0
#         for j in range(st):
#             k = tlags[j]
#             ik = i + k
#
#             # if |tlags| > 1, the prediction will be longer than yf
#             if ik < ny:
#                 yf[ik] = y[0, c:c + my]
#                 c += my
#
#         return i + lmax(tlags) + 1
#
#     def _update_temporal(self, i, y):
#         tlags = self._tlags
#         yf = self.yf
#
#         st = len(tlags)
#         ny = len(yf)
#
#         for j in range(st):
#             k = tlags[j]
#             ik = i + k
#
#             # if |tlags| > 1, the prediction will be longer than yf
#             if ik < ny:
#                 self.yf[ik] = y[0, j, :]
#
#         return i + lmax(tlags) + 1
#
#     def _update_channels(self, i, y):
#         tlags = self._tlags
#         yf = self.yf
#
#         st = len(tlags)
#         ny = len(yf)
#
#         for j in range(st):
#             k = tlags[j]
#             ik = i + k
#
#             # if |tlags| > 1, the prediction will be longer than yf
#             if ik < ny:
#                 self.yf[ik] = y[0, :, j]
#
#         return i + lmax(tlags) + 1
#
#     def _advance(self) -> int:
#         tlags = self._tlags
#         st = len(tlags)
#
#         for i in range(st):
#             if tlags[i] != i:
#                 return i
#         else:
#             return lmax(tlags) + 1
# # end
#
#
# # ---------------------------------------------------------------------------
# # LagsArrayTransformer
# # ---------------------------------------------------------------------------
#
# class LagsArrayTransformer(XyBaseEncoder):
#
#     def __init__(self,
#                  xlags=None,
#                  ylags=None,
#                  tlags=None,
#                  dtype=np.float32,
#                  temporal=False,
#                  channels=False,
#                  y_flatten=False):
#         super().__init__(None, False)
#
#         assert dtype is not None, "Parameter 'dtype' must be not None"
#         assert (not temporal and not channels
#                 or temporal and not channels
#                 or not temporal and channels), \
#             "Only a parameter between 'temporal' and 'channels' can be True"
#
#         self.xlags = xlags
#         self.ylags = ylags
#         self.tlags = tlags
#
#         self.dtype = dtype
#         self.temporal = temporal
#         self.channels = channels
#         self.y_flatten = y_flatten
#
#         self._xlags = _resolve_xylags(xlags)
#         self._ylags = _resolve_xylags(ylags)
#         self._tlags = _resolve_xylags(tlags, True)
#
#         self.X = None
#         self.y = None
#
#         self.Xh = None      # X history (ndarray)
#         self.yh = None      # y history (ndarray)
#         self.Xf = None      # X forecast (ndarray)
#         self.yf = None      # y forecast (ndarray)
#         self.Xp = None      # X prediction (ndarray)
#
#         self.Xp_shape = None    # X shape used in prediction
#         self.yp_shape = None    # y shape used in prediction
#         self.yf_shape = None    # y forecast shape
#
#     # -----------------------------------------------------------------------
#
#     def fit(self, X=None, y=None):
#         X, y = self._check_Xy(X, y)
#
#         self.X = X
#         self.y = y
#
#         X_ = _to_numpy(X, self.dtype)
#         y_ = _to_numpy(y, self.dtype)
#
#         self.Xh = X_
#         self.yh = y_
#
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#         st = len(tlags)
#
#         mx = X_.shape[1] if sx > 0 else 0  # n of features for x
#         my = y_.shape[1] if sy > 0 else 0  # n of features for y
#
#         if self.temporal:
#             nf = sx if sx > 0 else sy
#             self.Xp_shape = (1, nf, mx + my)
#             self.yp_shape = (1, st, my)
#         elif self.channels:
#             nf = sx if sx > 0 else sy
#             self.Xp_shape = (1, mx + my, nf)
#             self.yp_shape = (1, my, st)
#         else:
#             self.Xp_shape = (1, sx * mx + sy * my)
#             self.yp_shape = (1, st * my)
#
#         return self
#
#     def transform(self, X=None, y=None):
#         if isinstance(X, pd.PeriodIndex):
#             fh = X
#             X = pd.DataFrame(index=fh)
#
#         X, y = self._check_Xy(X, y)
#
#         X_ = _to_numpy(X, self.dtype)
#         y_ = _to_numpy(y, self.dtype)
#
#         self.Xf = X_
#         self.yf = y_
#
#         ih = self.y.index
#         if ih[0] == X.index[0]:
#             Xt, yt = self._transform_fitted(X_, y_)
#             xt = self._index_fitted(X.index)
#         elif ih[-1] == (X.index[0]-1):
#             Xt, yt = self._transform_forecast(X_, y_)
#             xt = self._index_forecast(X.index)
#         else:
#             raise ValueError("Invalid timestamps")
#
#         if self.y_flatten:
#             ny = yt.shape[0]
#             yt = yt.reshape((ny, -1))
#
#         return Xt, yt, xt
#
#     # -----------------------------------------------------------------------
#
#     def forecaster(self) -> LagsArrayForecaster:
#         laf = LagsArrayForecaster(
#             xlags=self.xlags,
#             ylags=self.ylags,
#             tlags=self.tlags,
#             dtype=self.dtype,
#             temporal=self.temporal,
#             channels=self.channels,
#             y_flatten=self.y_flatten
#         )
#
#         laf.fit(self.X, self.y)
#         return laf
#
#     # -----------------------------------------------------------------------
#
#     def _atx(self, i):
#         return self.Xh[i] if i < 0 else self.Xf[i]
#
#     def _aty(self, i):
#         return self.yh[i] if i < 0 else self.yf[i]
#
#     def _transform_fitted(self, X, y):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         s = max(lmax(xlags), lmax(ylags))
#
#         if self.temporal:
#             return self._transform_temporal(X, y, s)
#         elif self.channels:
#             return self._transform_channels(X, y, s)
#         else:
#             return self._transform_flatten(X, y, s)
#
#     def _transform_forecast(self, X, y):
#         if self.temporal:
#             return self._transform_temporal(X, y, 0)
#         elif self.channels:
#             return self._transform_channels(X, y, 0)
#         else:
#             return self._transform_flatten(X, y, 0)
#
#     # -----------------------------------------------------------------------
#
#     def _transform_flatten(self, X, y, s):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#         st = len(tlags)  # n of lags for t
#         t = lmax(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if sx > 0 else 0  # n of features for x
#         my = y.shape[1] if sy > 0 else 0  # n of features for y
#         mt = y.shape[1]
#         nl = n - (t + s)  # n of predicted rows
#
#         Xt = np.zeros((nl, sx*mx + sy*my), dtype=self.dtype)
#         yt = np.zeros((nl, st*mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(nl):
#
#             c = 0
#             for i in range(sx):
#                 k = xlags[i]
#                 ik = s + j - k
#                 Xt[j, c:c + mx] = atx(ik)
#                 c += mx
#
#             for i in range(sy):
#                 k = ylags[i]
#                 ik = s + j - k
#                 Xt[j, c:c + my] = aty(ik)
#                 c += my
#
#             c = 0
#             for i in range(st):
#                 k = tlags[i]
#                 ik = s + j + k
#                 yt[j, c:c + mt] = aty(ik)
#                 c += mt
#
#         return Xt, yt
#
#     def _transform_channels(self, X, y, s):
#         Xt, yt = self._transform_temporal(X, y, s)
#         Xt = Xt.swapaxes(1, 2)
#         return Xt, yt
#
#     def _transform_temporal(self, X, y, s):
#         xlags = list(reversed(self._xlags)) if X is not None else []
#         ylags = list(reversed(self._ylags)) if y is not None else []
#         tlags = self._tlags
#
#         n = len(X)  # n of rows
#         sx = len(xlags)  # n of lags for x
#         sy = len(ylags)  # n of lags for y
#         st = len(tlags)  # n of lags for t)arget
#         sf = max(sx, sy)  # n of lags for both features
#
#         assert sx == 0 or sy == 0 or sx == sy and xlags == ylags, "Valid x/y lags : [n, 0], [0, n], [n, n]"
#
#         t = lmax(tlags)  # window length for t)arget
#
#         mx = X.shape[1] if sx > 0 else 0  # n of features
#         my = y.shape[1] if sy > 0 else 0  # n of features
#         mt = y.shape[1]
#         nl = n - (s + t)  # n of predicted rows
#
#         Xt = np.zeros((nl, sf, mx + my), dtype=self.dtype)
#         yt = np.zeros((nl, st, mt), dtype=self.dtype)
#
#         atx = self._atx
#         aty = self._aty
#
#         for j in range(nl):
#
#             for i in range(sx):
#                 k = xlags[i]
#                 ik = s + j - k
#                 Xt[j, i, :mx] = atx(ik)
#
#             for i in range(sy):
#                 k = ylags[i]
#                 ik = s + j - k
#                 Xt[j, i, mx:] = aty(ik)
#
#             for i in range(st):
#                 k = tlags[i]
#                 ik = s + j + k
#                 yt[j, i, :] = aty(ik)
#
#         return Xt, yt
#
#     # -----------------------------------------------------------------------
#
#     def _index_fitted(self, ix):
#         xlags = list(reversed(self._xlags)) if self.Xh is not None else []
#         ylags = list(reversed(self._ylags)) if self.yh is not None else []
#         tlags = self._tlags
#
#         s = max(lmax(xlags), lmax(ylags))
#         t = lmax(tlags)
#         n = len(ix) - (s+t)
#
#         return ix[s:s+n]
#
#     def _index_forecast(self, ix):
#         tlags = self._tlags
#         st = lmax(tlags)
#         n = len(ix) - st
#
#         return ix[0:n]
# # end
#
#
# # ArrayTransformer = LagsArrayTransformer
#
#
