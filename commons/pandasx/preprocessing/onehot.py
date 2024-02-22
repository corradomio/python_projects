import numpy as np
from pandas import DataFrame

from .base import BaseEncoder
from ..base import safe_sorted


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

def inverse_map(map):
    imap = {}
    for k in map:
        v = map[k]
        imap[v] = k
    return imap


class OneHotEncoder(BaseEncoder):
    """
    OneHot encoding of the columns. If a column contains 1 or 2 values, it is
    converted in a single column
    """

    def __init__(self, columns=None, copy=True):
        super().__init__(columns, copy)
        self._maps = {}
        self._bins = {}

    def fit(self, X: DataFrame) -> "OneHotEncoder":
        X = self._check_X(X)

        for col in self._get_columns(X):
            values = safe_sorted(X[col].unique())
            n = len(values)

            # single value in [0,1]
            if n == 1 and values[0] in [0, 1]:
                continue
            # two values in [0,1]
            if n == 2 and values == [0, 1]:
                continue

            if n > 2:
                self._maps[col] = values
            else:
                self._bins[col] = values
        # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        # handle binary columns
        for col in self._bins:
            values = self._bins[col]
            vmap = {v: i for i, v in enumerate(values)}
            X.replace({col: vmap}, inplace=True)

        # handle the other columns
        for col in self._maps:
            values = self._maps[col]

            # prepare the columns
            for v in values:
                ohcol = f"{col}_{v}"
                X.insert(len(X.columns), ohcol, 0)
                X[ohcol] = X[ohcol].astype(np.int8)

            # fill the columns
            for v in values:
                ohcol = f"{col}_{v}"
                X.loc[X[col] == v, ohcol] = 1
        # end

        # remove the columns
        X.drop(self._maps.keys(), axis=1, inplace=True)
        return X

    def inverse_transform(self, X:DataFrame) -> DataFrame:
        # reverse the binary columns
        for col in self._bins:
            values = self._bins[col]
            imap = {i: v for i, v in enumerate(values)}
            X.replace({col: imap}, inplace=True)

        # reverse the other columns
        ohcols = []
        for col in self._maps:
            values = self._maps[col]

            # prepare the column filling with the first value
            X[col] = values[0]

            # scan the values:
            for v in values:
                ohcol = f"{col}_{v}"
                X.loc[X[ohcol] == 1, col] = v

                ohcols.append(ohcol)
            pass
        # end

        # drop the columns
        X.drop(ohcols, axis=1, inplace=True)
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
