import numpy as np
import pandas as pd

from .base import GroupsBaseEncoder


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------

# def _oversample2d(a: np.ndarray, nsamples=10) -> np.ndarray:
#     """
#
#     """
#     assert isinstance(a, np.ndarray) and len(a.shape) == 2, \
#         "Parameter 'a' is not a 2D numpy array"
#
#     n0, m = a.shape
#     ns = nsamples * (n0 - 1) + 1
#     resampled = np.zeros((ns, m), dtype=a.dtype)
#
#     j = 0
#     for i in range(n0-1):
#         y0 = a[i + 0]
#         y1 = a[i + 1]
#         dy = (y1 - y0) / nsamples
#         for k in range(nsamples):
#             resampled[j] = y0 + k * dy
#             j += 1
#         # end
#     # end
#     resampled[-1] = a[-1]
#
#     return resampled
# # end
#
#
# def _undersample2d(a: np.ndarray, nsamples=10) -> np.ndarray:
#     assert isinstance(a, np.ndarray) and len(a.shape) == 2, \
#         "Parameter 'a' is not a 2D numpy array"
#
#     #nsamples * (n0 - 1) + 1
#
#     n0, m = a.shape
#     ns = (n0 - 1)//nsamples + 1
#     resampled = np.zeros((ns, m), dtype=a.dtype)
#
#     j = 0
#     for i in range(0, n0, nsamples):
#         resampled[j] = a[i]
#         j += 1
#     resampled[-1] = a[-1]
#     return resampled
# # end
#
#
# class Resampler(GroupsBaseEncoder):
#
#     def __init__(self, nsamples=10, columns=None, groups=None, copy=True):
#         super().__init__(columns, groups, copy)
#         self.nsamples = nsamples
#
#     def _get_params(self, g):
#         return None
#
#     def _set_params(self, g, params):
#         return
#
#     def _compute_params(self, g, X: pd.DataFrame):
#         return None
#
#     def _apply_transform(self, X: pd.DataFrame, params):
#         if self.nsamples <= 1:
#             return X
#
#         data: np.ndarray = X.values
#         resampled = _oversample2d(data, self.nsamples)
#         n = len(resampled)
#
#         idx_s: pd.Period = X.index[ 0]
#         idx_e: pd.Period = X.index[-1]
#         index = pd.date_range(idx_s.start_time, idx_e.end_time, periods=n)
#
#         return pd.DataFrame(data=resampled, columns=X.columns, index=index)
#         pass
#
#     def _apply_inverse_transform(self, X, params):
#         if self.nsamples <= 1:
#             return X
#
#         data = X.values
#         resampled = _undersample2d(data, self.nsamples)
#         n = len(resampled)
#
#         idx_s: pd.Period = X.index[ 0]
#         idx_e: pd.Period = X.index[-1]
#         index = pd.date_range(idx_s.start_time, idx_e.end_time, periods=n)
#
#         return pd.DataFrame(data=resampled, columns=X.columns, index=index)
#         pass
# # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
