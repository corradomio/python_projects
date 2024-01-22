# from typing import Union, Optional
#
# import numpy as np
# import pandas as pd
# from sktime.forecasting.base import ForecastingHorizon
#
#
# # ---------------------------------------------------------------------------
# # Utilities
# # ---------------------------------------------------------------------------
#
# NoneType = type(None)
# RangeType = type(range(0))
# ARRAY_OR_DF = Union[NoneType, np.ndarray, pd.DataFrame]
#
#
# def lmax(l: list[int]) -> int:
#     return 0 if len(l) == 0 else max(l)
#
#
# def lrange(start, stop=None, step=None):
#     """
#     Same as 'list(range(...))'
#     """
#     if stop is None:
#         return list(range(start))
#     elif step is None:
#         return list(range(start, stop))
#     else:
#         return list(range(start, stop, step))
#
#
# def to_matrix(data: Union[NoneType, pd.Series, pd.DataFrame, np.ndarray], dtype=np.float32) -> Optional[np.ndarray]:
#     if data is None:
#         return None
#     if isinstance(data, pd.Series):
#         data = data.to_numpy().astype(dtype).reshape((-1, 1))
#     elif isinstance(data, pd.DataFrame):
#         data = data.to_numpy().astype(dtype)
#     elif len(data.shape) == 1:
#         assert isinstance(data, np.ndarray)
#         data = data.astype(dtype).reshape((-1, 1))
#     # elif isinstance(data, np.ndarray) and dtype is not None and data.dtype != dtype:
#     #     data = data.astype(dtype)
#     elif isinstance(data, np.ndarray):
#         pass
#     return data
#
#
# # ---------------------------------------------------------------------------
# # Lags
# # ---------------------------------------------------------------------------
#
# def resolve_ilags(ilags: Union[int, tuple, list]) -> list[int]:
#     """
#     Resolve i)input lags (xlags & ylags).
#     Note: the list must be ordered in increase way BUT processed in the
#           opposite way
#     """
#     if isinstance(ilags, int):
#         # [1, ilags]
#         return list(range(1, ilags+1))
#     elif isinstance(ilags, RangeType):
#         return sorted(list(ilags))
#     else:
#         assert isinstance(ilags, (list, tuple))
#         return sorted(list(ilags))
#
#
# def resolve_tlags(tlags: Union[int, tuple, list]) -> list[int]:
#     """
#     Resolve t)arget lags.
#     """
#     if isinstance(tlags, int):
#         # [0, tlags-1]
#         return list(range(tlags))
#     elif isinstance(tlags, RangeType):
#         return sorted(list(tlags))
#     else:
#         assert isinstance(tlags, (list, tuple))
#         return sorted(list(tlags))
#
#
# # ---------------------------------------------------------------------------
# #   ModelTransform
# #       ModelTrainTransform
# #       ModePredictTransform
# # ---------------------------------------------------------------------------
#
# class TimeseriesTransform:
#
#     def _check_X(self, X):
#         X = to_matrix(X)
#         assert isinstance(X, (NoneType, np.ndarray))
#         return X
#
#     def _check_y(self, y):
#         y = to_matrix(y)
#         assert isinstance(y, np.ndarray)
#         return y
#
#     def _check_Xy(self, X, y=None, fh=None):
#         X = to_matrix(X)
#         y = to_matrix(y)
#         assert fh is None
#         assert isinstance(X, (NoneType, np.ndarray))
#         assert isinstance(y, (NoneType, np.ndarray))
#         return X, y
#
#     def _check_Xfh(self, X, fh, y):
#         assert y is None
#         # Note: to be compatible with 'sktime' fh MUST starts 1 timeslot after the
#         # 'cutoff', that is, the LAST timestamp used in training. This means that
#         # If specified as list of as a ForecastingHorizon, it MUST be:
#         #
#         #   [1,2,3.....]
#         #
#         # Then, the LAST value is the 'prediction_length'.
#         #
#         X = to_matrix(X)
#         if fh is None or fh <= 0:
#             assert X is not None, "If fh is not specified, X must be not None"
#             fh = len(X)
#         elif isinstance(fh, list):
#             assert len(fh) > 0 and fh[0] >= 1, f'fh can not start with a value < 1 with ({fh[0]})'
#             fh = fh[-1]
#         elif isinstance(fh, ForecastingHorizon):
#             assert fh.is_relative, f'fh must be a relative ForecastingHorizon'
#             assert len(fh) > 0 and fh[0] >= 1, f'fh can not start with a value < 1 with ({fh[0]})'
#             fh = fh[-1]
#
#         assert isinstance(X, (NoneType, np.ndarray))
#         assert isinstance(fh, int)
#         return X, fh
#
#     def fit(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None) -> "TimeseriesTransform":
#         return self
#
#     def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF = None, fh=None) -> tuple:
#         return X, y
#
#     def fit_transform(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF=None, fh=None):
#         return self.fit(y=y, X=X).transform(y=y, X=X)
# # end
#
#
# class ModelTrainTransform(TimeseriesTransform):
#
#     def __init__(self, slots, tlags):
#         assert isinstance(slots, (list, tuple)) and len(slots) == 2
#         xlags = resolve_ilags(slots[0])
#         ylags = resolve_ilags(slots[1])
#         tlags = resolve_tlags(tlags)
#
#         self.slots = slots
#         self.xlags: list = xlags
#         self.ylags: list = ylags
#         self.tlags: list = tlags
#     # end
#
#     # def fit(self, y: np.ndarray, X: Optional[np.ndarray]=None):
#     #     return self
#     # # end
#
#     # def transform(self, y: np.ndarray, X: Optional[np.ndarray]=None) -> tuple:
#     #     return X, y
#     # # end
#
#     # def fit_transform(self, y: np.ndarray, X: Optional[np.ndarray]=None):
#     #     return self.fit(y=y, X=X).transform(y=y, X=X)
# # end
#
#
# class ModelPredictTransform(TimeseriesTransform):
#
#     def __init__(self, slots, tlags):
#         assert isinstance(slots, (list, tuple)) and len(slots) == 2
#         xlags = resolve_ilags(slots[0])
#         ylags = resolve_ilags(slots[1])
#         tlags = resolve_tlags(tlags)
#
#         self.slots = slots
#         self.xlags: list[int] = xlags
#         self.ylags: list[int] = ylags
#         self.tlags: list[int] = tlags
#
#         self.Xh = None  # X history
#         self.yh = None  # y history
#
#         self.Xt = None  # X transform
#         self.yt = None  # y transform
#         self.fh = None  # fh transform
#
#         self.Xp = None  # X prediction past
#         self.yy = None  # y prediction past
#         self.Xf = None  # X prediction future
#         self.yp = None  # y prediction future
#     # end
#
#     def fit(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None):
#         # used to support the implementation of 'self.to_pandas()'
#         self.y_pandas = y
#
#         # This method must be used to pass 'y_train' and 'X_train'
#         X, y = self._check_Xy(X, y)
#
#         self.Xh = X
#         self.yh = y
#
#         return self
#     # end
#
#     def transform(self, fh: int, X: ARRAY_OR_DF = None, y=None):
#         # This method must e used with 'fh' and 'X_test'/'X_predict'
#         X, fh = self._check_Xfh(X, fh, y)
#
#         self.Xt = X
#         self.fh = fh
#
#         return fh, X
#     # end
#
#     def update(self, i, y_pred, t=None):
#         tlags = [t] if t is not None else self.tlags
#         st = len(tlags)
#         mt = lmax(tlags)
#         nfh = len(self.yp)
#         my = self.yp.shape[1]
#
#         if len(y_pred.shape) == 1:
#             y_pred = y_pred.reshape((1, 1, my))
#         elif len(y_pred.shape) == 2:
#             y_pred = y_pred.reshape((1, -1, my))
#
#         assert len(y_pred.shape) == 3
#
#         for j in range(st):
#             k = i + tlags[j]
#             if k < nfh:
#                 self.yp[k] = y_pred[0, j]
#
#         return i + mt + 1
#     # end
#
#     def fit_transform(self, y, X=None):
#         # It is not correct to use 'fit_transform(y, X)' because
#         # 1) with 'fit(y,X)' it is passed y_train and X_train
#         # 2) with 'transform(fh, X)' it is passed the forecasting horizon and X_prediction
#         # that is, different information
#         raise NotImplemented()
#
#     def to_pandas(self, yp):
#         if not isinstance(self.y_pandas, (pd.DataFrame, pd.Series)):
#             return yp
#
#         n = len(yp)
#         cutoff = self.y_pandas.index[-1]
#         fh = ForecastingHorizon(lrange(1, n+1), is_relative=True).to_absolute(cutoff)
#         if isinstance(self.y_pandas, pd.DataFrame):
#             return pd.DataFrame(data=yp, columns=self.y_pandas.columns, index=fh.to_pandas())
#         else:
#             return pd.Series(data=yp, name=self.y_pandas.name, index=fh.to_pandas())
# # end
