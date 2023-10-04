import numpy as np
from pandas import DataFrame, RangeIndex

from .base import XyBaseEncoder


#
# window lags
#
#   lags = None     == [[0], []]
#   lags = n        == [n, n]
#   lags = []       == [[0], []]
#   lags = [n]      == [n, []]
#   lags = [n, m]
#   lags = [[...]]  == [[...], []]
#   lags = [[...], [...]]
#
#   n is converted in [1...n]
#
# forecast lags
#
#   lags = None     == [0]
#   lags = n        == [0...n-1]
#   lags = []       == []
#   lags = [n]
#   lags = [[...]]
#
#   n is converted in [0...n-1]
#

def _resolve_xylags(lags, forecast=False):
    if lags is None and forecast:
        return [0]
    elif lags is None:
        return []
    elif isinstance(lags, int):
        s = 0 if forecast else 1
        return list(range(s, s+lags))
    else:
        return lags


# def _resolve_lags(lags, forecast=False):
#     if forecast:
#         if lags is None:
#             return []
#         elif isinstance(lags, int):
#             return list(range(lags))
#         else:
#             return lags
#     # end
#     if lags is None:
#         lags = [[0], []]
#     elif isinstance(lags, int):
#         lags = [lags, lags]
#     if len(lags) == 0:
#         lags = [[0], []]
#     elif len(lags) == 1:
#         lags = lags + [[]]
#
#     xlags, ylags = lags
#     if isinstance(xlags, int):
#         xlags = list(range(1, xlags+1))
#     if isinstance(ylags, int):
#         ylags = list(range(1, ylags+1))
#
#     return xlags, ylags

def lmax(l): return max(l) if l else 0


def _add_col(Xt, X, col, lags, s, n, forecast=False, Xh=None):
    def name_of(k):
        if k < 0:
            return f"{col}{k:02}"
        elif k > 0:
            return f"{col}_{k:02}"
        else:
            return col

    if not forecast:
        lags = reversed(lags)

    for k in lags:
        k = k if forecast else -k
        b = s + k
        if b < 0:
            b = -b
            h = Xh[col].to_numpy()
            x = X[col].to_numpy()
            f = np.zeros(n, dtype=h.dtype)
            f[0:b] = h[-b:]
            f[b:] = x[0:n - b]
        else:
            x = X[col].to_numpy()
            f = x[b:n + b]
        fname = name_of(k)
        Xt[fname] = f
    return Xt


# ---------------------------------------------------------------------------
# LagsTransformer
# ---------------------------------------------------------------------------
# Note: it is not possible to join X and y because the time stamps are
# different.
#
# NO: it is useless to apply a LagTransformer to a multi time-series
#

class LagsTransformer(XyBaseEncoder):
    """
    Add a list of columns based on xlags, ylags and tlags, where:

        - xlags: lags applied to columns
        - ylags: lags applied to target(s)
        - tlags: lags applied to target(s) and used for forecasting

    xlags and ylags are used to compose X, tlags is used to compose y
    """

    def __init__(self,
                 xlags=None,
                 ylags=None,
                 tlags=None):
        """

        :param columns: columns to use as features
        :param target: column(s) to use ad target
        :param lags: lags used on the train features
        :param copy:
        """
        super().__init__(None, False)
        self.xlags = xlags
        self.ylags = ylags
        self.tlags = tlags

        self._xlags = _resolve_xylags(xlags)
        self._ylags = _resolve_xylags(ylags)
        self._tlags = _resolve_xylags(tlags, True)

        self.Xh = None
        self.yh = None
        self.Xf = None
        self.yf = None

    def fit(self, X, y):
        super().fit(X, y)
        self.Xh = X
        self.yh = y
        return self

    def transform(self, X: DataFrame, y: DataFrame):
        X, y = self._check_Xy(X, y)

        if self.Xh.index[0] == X.index[0]:
            return self._transform_fitted(X, y)
        else:
            return self._transform_forecast(X, y)

    def _transform_fitted(self, X, y):
        assert self.Xh.index[0] == X.index[0]
        assert self.Xh.index[-1] == X.index[-1]

        xlags = self._xlags if X is not None else []
        ylags = self._ylags
        tlags = self._tlags

        s = max(lmax(xlags), lmax(ylags))
        st = max(tlags)
        n = len(X) - (s + st)
        ix = y.index

        Xt = DataFrame(index=RangeIndex(n))
        for col in X.columns:
            _add_col(Xt, X, col, xlags, s, n)

        for col in y.columns:
            _add_col(Xt, y, col, ylags, s, n)

        Xt.index = ix[s:s+n]

        yt = DataFrame(index=RangeIndex(n))
        for col in y.columns:
            _add_col(yt, y, col, tlags, s, n, True)

        yt.index = ix[s:s+n]

        return Xt, yt

    def _transform_forecast(self, X, y):
        # apply the transformation using Xh, yh
        assert self.Xh.index[-1] == (X.index[0]-1)

        xlags = self._xlags if X is not None else []
        ylags = self._ylags
        tlags = self._tlags
        st = max(tlags)

        n = len(y)
        ix = y.index - st

        Xt = DataFrame(index=RangeIndex(n))
        for col in X.columns:
            _add_col(Xt, X, col, xlags, -st, n, False, self.Xh)

        for col in y.columns:
            _add_col(Xt, y, col, ylags, -st, n, False, self.yh)

        Xt.index = ix

        yt = DataFrame(index=RangeIndex(n))
        for col in y.columns:
            _add_col(yt, y, col, tlags, -st, n, True, self.yh)

        yt.index = ix

        return Xt, yt
# end


# class LagsTransformer(BaseEncoder):
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
#     def __init__(self, columns=None,
#                  target=None,
#                  xlags=None,
#                  ylags=None,
#                  copy=True):
#         """
#
#         :param columns: columns to use as features
#         :param target: column(s) to use ad target
#         :param lags: lags used on the train features
#         :param copy:
#         """
#         super().__init__(columns, copy)
#         self.targets = as_list(target)
#         self.xlags = xlags
#         self.ylags = ylags
#
#         self._xlags = _resolve_xylags(xlags)
#         self._ylags = _resolve_xylags(ylags, True)
#
#     def transform(self, df: DataFrame) -> (DataFrame, DataFrame):
#         """
#         Apply the lags transformations.
#         If tlags is not defined, it is not composed y
#
#         :param df:
#         :return: (X, y) if tlags is defined otherwise (X, None)
#         """
#         df = self._check_X(df)
#
#         xlags = self._xlags
#         ylags = self._ylags
#         s = _window_len(xlags, ylags)
#         n = len(df) - s
#
#         columns = self._get_columns(df)
#         targets = self.targets
#         features = columns
#         ix = df.index
#
#         X = DataFrame(index=RangeIndex(n))
#         for col in features:
#             _add_col(X, df, col, xlags, s, n)
#
#         for col in targets:
#             _add_col(X, df, col, ylags, s, n)
#
#         X.index = ix[s:s+n]
#
#         return X
#     # end
# # end


# ---------------------------------------------------------------------------
# LagsTransformerOld
# ---------------------------------------------------------------------------
# Note: it is not possible to join X and y because the time stamps are
# different.
#

# class LagsTransformerOld(BaseEncoder):
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
#     def __init__(self, columns=None,
#                  target=None,
#                  lags=None, tlags=None,
#                  copy=True):
#         """
#
#         :param columns: columns to use as features
#         :param target: column(s) to use ad target
#         :param lags: lags used on the train features
#         :param tlags: lags used on the predicted targets
#         :param copy:
#         """
#         super().__init__(columns, copy)
#         self.targets = as_list(target)
#         self.lags = lags
#         self.tlags = tlags
#
#         self._slots = _resolve_lags(lags)
#         self._tlags = _resolve_xylags(tlags)
#
#     def transform(self, df: DataFrame) -> (DataFrame, DataFrame):
#         """
#         Apply the lags transformations.
#         If tlags is not defined, it is not composed y
#
#         :param df:
#         :return: (X, y) if tlags is defined otherwise (X, None)
#         """
#         df = self._check_X(df)
#
#         xlags, ylags = self._slots
#         tlags = self._tlags
#
#         w = _window_len(xlags, ylags, tlags)
#         s = _window_len(xlags, ylags)
#         n = len(df) - w
#
#         columns = self._get_columns(df)
#         targets = self.targets
#         features = list(set(columns).difference(targets))
#         ix = df.index
#
#         X = DataFrame(index=RangeIndex(n))
#         for col in features:
#             _add_col(X, df, col, xlags, s, n)
#
#         for col in targets:
#             _add_col(X, df, col, ylags, s, n)
#
#         X.index = ix[s:s+n]
#
#         # if tlags is None or [], return ONLY X
#         if not tlags:
#             return X, None
#
#         y = DataFrame(index=RangeIndex(n))
#         for col in targets:
#             _add_col(y, df, col, tlags, s, n, True)
#
#         y.index = ix[w:w+n]
#
#         return X, y
#     # end
# # end
