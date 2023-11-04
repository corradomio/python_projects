from numpy import issubdtype, integer
from pandas import DataFrame
from .base import BaseEncoder


# ---------------------------------------------------------------------------
# BinaryLabelsEncoder
# ---------------------------------------------------------------------------

class BinaryLabelsEncoder(BaseEncoder):
    """
    Convert not integer values in the column in {0,1}
    It is applied to columns with 1 or 2 not integer values ONLY
    Note: OneHotEncoder is able to encode binary columns in the correct way
    """

    def __init__(self, columns=None, copy=True):
        super().__init__(columns, copy)
        self._maps = {}

    def fit(self, X: DataFrame) -> "BinaryLabelsEncoder":
        X = self._check_X(X)

        for col in self._get_columns(X):
            x = X[col]
            if issubdtype(x.dtype.type, integer):
                continue

            values = list(sorted(x.unique()))
            nv = len(values)

            if nv == 1 and values[0] in [0, 1]:     # skip if a single value in [0, 1]
                continue
            if values == [0, 1]:                    # skip if values in [0, 1]
                continue
            if nv > 2:                              # skip if there are 3+ values
                continue

            if nv == 1:                             # v -> 0
                v = values[0]
                vmap = {v: 0}
            else:                                   # u -> 0, v -> 1
                u, v = values
                vmap = {u: 0, v: 1}

            self._maps[col] = vmap
        # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        X.replace(self._maps, inplace=True)

        # for col in self._get_columns(X):
        #     if col not in self._maps:
        #         continue
        #
        #     vmap = self._maps[col]
        #     X.replace({col: vmap}, inplace=True)
        # # end
        return X
# end


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

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
            x = X[col]

            values = list(sorted(x.unique()))
            n = len(values)

            if n == 1 and values[0] in [0, 1]:
                continue
            if n == 2 and values == [0, 1]:
                continue

            vmap = {}
            for i in range(n):
                v = values[i]
                vmap[v] = i

            if n > 2:
                self._maps[col] = vmap
            else:
                self._bins[col] = vmap

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        for col in self.columns:
            if col in self._maps:
                vmap = self._maps[col]

                for v in vmap:
                    vcol = f"{col}_{v}"
                    X[vcol] = 0
                    X.loc[X[col] == v, vcol] = 1

            if col in self._bins:
                vmap = self._bins[col]
                X.replace({col: vmap}, inplace=True)
        # end

        return X
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
