from stdlib import as_list
from pandas import DataFrame, RangeIndex
from .base import BaseEncoder


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
    if lags is None:
        return []
    elif isinstance(lags, int):
        s = 1 if forecast else 0
        return list(range(s, s+lags))
    else:
        return lags


def _resolve_lags(lags, forecast=False):
    if forecast:
        if lags is None:
            return []
        elif isinstance(lags, int):
            return list(range(lags))
        else:
            return lags
    # end
    if lags is None:
        lags = [[0], []]
    elif isinstance(lags, int):
        lags = [lags, lags]
    if len(lags) == 0:
        lags = [[0], []]
    elif len(lags) == 1:
        lags = lags + [[]]

    xlags, ylags = lags
    if isinstance(xlags, int):
        xlags = list(range(1, xlags+1))
    if isinstance(ylags, int):
        ylags = list(range(1, ylags+1))

    return xlags, ylags


def _window_len(xlags: list, ylags=None, tlags=None) -> int:
    def maxl(l) -> int: return max(l) if l else 0

    xylen = max(maxl(xlags), maxl(ylags))
    tlen = maxl(tlags)
    return xylen + tlen


def _add_col(X, df, col, lags, s, n, forecast=False):
    def name_of(k):
        if k < 0:
            return f"{col}{k:02}"
        elif k > 0:
            return f"{col}_{k:02}"
        else:
            return col

    if lags == [1] or lags == [0]:
        k = lags[0]
        k = k if forecast else -k
        b = s + k
        f = df[col].iloc[b:b + n].reset_index()[col]
        X[col] = f
        return X

    if not forecast:
        lags = reversed(lags)

    for k in lags:
        k = k if forecast else -k
        b = s + k
        f = df[col].iloc[b:b + n].reset_index()[col]
        # fname = _name_of(col, k)
        fname = name_of(k)
        X[fname] = f
    return X


# ---------------------------------------------------------------------------
# LagsTransformer
# ---------------------------------------------------------------------------
# Note: it is not possible to join X and y because the time stamps are
# different.
#

class LagsTransformer(BaseEncoder):
    """
    Add a list of columns based on xlags, ylags and tlags, where:

        - xlags: lags applied to columns
        - ylags: lags applied to target(s)
        - tlags: lags applied to target(s) and used for forecasting

    xlags and ylags are used to compose X, tlags is used to compose y
    """

    def __init__(self, columns=None,
                 target=None,
                 xlags=None,
                 ylags=None,
                 copy=True):
        """

        :param columns: columns to use as features
        :param target: column(s) to use ad target
        :param lags: lags used on the train features
        :param copy:
        """
        super().__init__(columns, copy)
        self.targets = as_list(target)
        self.xlags = xlags
        self.ylags = ylags

        self._xlags = _resolve_xylags(xlags)
        self._ylags = _resolve_xylags(ylags, True)

    def transform(self, df: DataFrame) -> (DataFrame, DataFrame):
        """
        Apply the lags transformations.
        If tlags is not defined, it is not composed y

        :param df:
        :return: (X, y) if tlags is defined otherwise (X, None)
        """
        df = self._check_X(df)

        xlags = self._xlags
        ylags = self._ylags
        s = _window_len(xlags, ylags)
        n = len(df) - s

        columns = self._get_columns(df)
        targets = self.targets
        features = columns
        ix = df.index

        X = DataFrame(index=RangeIndex(n))
        for col in features:
            _add_col(X, df, col, xlags, s, n)

        for col in targets:
            _add_col(X, df, col, ylags, s, n)

        X.index = ix[s:s+n]

        return X
    # end
# end


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
