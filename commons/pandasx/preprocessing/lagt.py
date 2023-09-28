from stdlib import as_list
from pandas import DataFrame, RangeIndex
from .base import BaseEncoder


#
#   lags = n        == [n, n]
#   lags = [n]      == [n, 0]
#   lags = [n, m]
#   lags = [[...], [...]]

def _resolve_lags(lags, forecast=False):
    if forecast:
        if lags is None:
            return [0]
        elif isinstance(lags, int):
            return list(range(lags))
        else:
            return lags
    # end
    if lags is None:
        lags = [0, 0]
    elif isinstance(lags, int):
        lags = [lags, lags]
    if len(lags) == 0:
        lags = [0, 0]
    elif len(lags) == 1:
        lags = lags + [0]

    xlags, ylags = lags
    if isinstance(xlags, int):
        xlags = list(range(1, xlags+1))
    if isinstance(ylags, int):
        ylags = list(range(1, ylags+1))

    return xlags, ylags


def _window_len(xlags, ylags, tlags):
    xylen = max(max(xlags) if xlags != [] else 0, max(ylags) if ylags != [] else 0)
    tlen = max(tlags)
    return xylen + tlen


def _name_of(col, k):
    return f"{col}_{k:02}"


def _add_col(X, df, col, lags, s, n, forecast=False):
    if lags == [1] or lags == [0]:
        k = lags[0]
        k = k if forecast else -k
        b = s + k
        f = df[col].iloc[b:b + n].reset_index()[col]
        X[col] = f
        return X

    for k in reversed(lags):
        k = k if forecast else -k
        b = s + k
        f = df[col].iloc[b:b + n].reset_index()[col]
        fname = _name_of(col, k)
        X[fname] = f
    return X


# ---------------------------------------------------------------------------
# LagsTransformer
# ---------------------------------------------------------------------------

class LagsTransformer(BaseEncoder):

    def __init__(self, columns=None, target=None, lags=None, tlags=None, copy=True):
        super().__init__(columns, copy)
        self.targets = as_list(target)
        self.lags = lags
        self.tlags = tlags
        self._xcols = self.columns
        self._ycols = self.targets
        self._tlags = _resolve_lags(tlags, True)
        self._slots = _resolve_lags(lags)

    def transform(self, df: DataFrame) -> (DataFrame, DataFrame):
        xlags, ylags = self._slots
        tlags = self._tlags
        s = _window_len(xlags, ylags, tlags)
        n = len(df) - s

        columns = self._get_columns(df)
        targets = self.targets
        features = list(set(columns).difference(targets))
        ix = df.index

        X = DataFrame(index=RangeIndex(n))
        for col in features:
            _add_col(X, df, col, xlags, s, n)

        for col in targets:
            _add_col(X, df, col, ylags, s, n)

        X.index = ix[0:n]

        y = DataFrame(index=RangeIndex(n))

        for col in targets:
            _add_col(y, df, col, tlags, s, n, True)

        y.index = ix[s:]

        return X, y
    # end
# end
