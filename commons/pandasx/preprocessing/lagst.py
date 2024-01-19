# import numpy as np
# import pandas as pd
#
# from .base import XyBaseEncoder, as_list
#
#
# #
# # window lags
# #
# #   lags = None     == [[0], []]
# #   lags = n        == [n, n]
# #   lags = []       == [[0], []]
# #   lags = [n]      == [n, []]
# #   lags = [n, m]
# #   lags = [[...]]  == [[...], []]
# #   lags = [[...], [...]]
# #
# #   n is converted in [1...n]
# #
# # forecast lags
# #
# #   lags = None     == [0]
# #   lags = n        == [0...n-1]
# #   lags = []       == []
# #   lags = [n]
# #   lags = [[...]]
# #
# #   n is converted in [0...n-1]
# #
# RangeType = type(range(0))
#
#
# def _resolve_xylags(lags, forecast=False):
#     if lags is None and forecast:
#         return []
#         # return [0]
#     elif lags is None:
#         return []
#     elif isinstance(lags, int):
#         s = 0 if forecast else 1
#         return list(range(s, s+lags))
#     elif isinstance(lags, RangeType):
#         return list(lags)
#     elif isinstance(lags, (list, tuple)):
#         return lags
#     else:
#         raise ValueError(f"Unsupported lags '{lags}'")
#
#
# def lmax(l): return max(l) if l else 0
#
#
# def _add_col(Xt, X, col, lags, s, n, forecast=False, Xh=None):
#     def name_of(k):
#         if k < 0:
#             k = -k
#             return f"{col}-{k:02}"
#         elif k > 0:
#             return f"{col}+{k:02}"
#         else:
#             return col
#
#     if not forecast:
#         lags = reversed(lags)
#
#     for k in lags:
#         k = k if forecast else -k
#         b = s + k
#         if b < 0:
#             b = -b
#             h = Xh[col].to_numpy()
#             x = X[col].to_numpy()
#             f = np.zeros(n, dtype=h.dtype)
#             f[0:b] = h[-b:]
#             f[b:] = x[0:n - b]
#         else:
#             x = X[col].to_numpy()
#             f = x[b:n + b]
#         fname = name_of(k)
#         Xt[fname] = f
#     return Xt
#
#
# # ---------------------------------------------------------------------------
# # LagsTransformer
# # ---------------------------------------------------------------------------
# # Note: it is not possible to join X and y because the time stamps are
# # different.
# #
# # NO: it is useless to apply a LagTransformer to a multi time-series
# #
#
# class LagsTransformer(XyBaseEncoder):
#     """
#     Add a list of columns based on xlags, ylags and tlags, where:
#
#         - xlags: lags applied to columns
#         - ylags: lags applied to target(s)
#         - tlags: lags applied to target(s) and used for forecasting
#
#     xlags and ylags are used to compose X, tlags is used to compose y
#     """
#
#     def __init__(self,
#                  xlags=None,
#                  ylags=None,
#                  tlags=None,
#                  target=None):
#         """
#
#         :param columns: columns to use as features
#         :param target: column(s) to use ad target
#         :param lags: lags used on the train features
#         :param copy:
#         """
#         super().__init__(None, False)
#         self.xlags = xlags
#         self.ylags = ylags
#         self.tlags = tlags
#         self.target = as_list(target, "target")
#
#         self._xlags = _resolve_xylags(xlags)
#         self._ylags = _resolve_xylags(ylags)
#         self._tlags = _resolve_xylags(tlags, True)
#
#         self.Xh = None
#         self.yh = None
#         self.Xf = None
#         self.yf = None
#
#     def fit(self, X=None, y=None):
#         super().fit(X, y)
#
#         if len(self.target) > 0:
#             y = X[self.target]
#
#         X, y = self._check_Xy(X, y)
#
#         self.Xh = X
#         self.yh = y
#         return self
#
#     def transform(self, X=None, y=None):
#         if len(self.target) > 0:
#             y = X[self.target]
#         if X is None:
#             X = y.index
#         if isinstance(X, pd.PeriodIndex):
#             ix = X
#             X = pd.DataFrame(index=ix)
#
#         X, y = self._check_Xy(X, y)
#
#         if self.Xh.index[0] == X.index[0]:
#             Xt, yt = self._transform_fitted(X, y)
#         else:
#             Xt, yt = self._transform_forecast(X, y)
#
#         return (Xt, yt) if self.tlags else Xt
#
#     def _transform_fitted(self, X, y):
#         assert self.Xh.index[0] == X.index[0]
#         assert self.Xh.index[-1] == X.index[-1]
#
#         xlags = self._xlags if X is not None else []
#         ylags = self._ylags
#         tlags = self._tlags
#
#         s = max(lmax(xlags), lmax(ylags))
#         t = lmax(tlags)
#         n = len(X) - (s + t)
#         ix = y.index
#
#         Xt = pd.DataFrame(index=pd.RangeIndex(n))
#         for col in X.columns:
#             _add_col(Xt, X, col, xlags, s, n)
#
#         for col in y.columns:
#             _add_col(Xt, y, col, ylags, s, n)
#
#         Xt.index = ix[s:s+n]
#
#         yt = pd.DataFrame(index=pd.RangeIndex(n))
#         for col in y.columns:
#             _add_col(yt, y, col, tlags, s, n, True)
#
#         yt.index = ix[s:s+n]
#
#         return Xt, yt
#
#     def _transform_forecast(self, X, y):
#         # apply the transformation using Xh, yh
#         assert self.Xh.index[-1] == (X.index[0]-1)
#
#         xlags = self._xlags if X is not None else []
#         ylags = self._ylags
#         tlags = self._tlags
#         t = lmax(tlags)
#
#         n = len(y)
#         ix = y.index - t
#
#         Xt = pd.DataFrame(index=pd.RangeIndex(n))
#         for col in X.columns:
#             _add_col(Xt, X, col, xlags, -t, n, False, self.Xh)
#
#         for col in y.columns:
#             _add_col(Xt, y, col, ylags, -t, n, False, self.yh)
#
#         Xt.index = ix
#
#         yt = pd.DataFrame(index=pd.RangeIndex(n))
#         for col in y.columns:
#             _add_col(yt, y, col, tlags, -t, n, True, self.yh)
#
#         yt.index = ix
#
#         return Xt, yt
# # end
#
#
# # ---------------------------------------------------------------------------
# # End
# # ---------------------------------------------------------------------------
#
