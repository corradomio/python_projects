from numpy import issubdtype, integer

from .base import *


# ---------------------------------------------------------------------------
# BinaryLabelsEncoder
# ---------------------------------------------------------------------------

class BinaryLabelsEncoder(BaseEncoder):
    """
    Convert the values in the column in {0,1}
    """

    def __init__(self, columns, copy=True):
        super().__init__(columns, copy)
        self._maps = {}

    def fit(self, X: DataFrame) -> "BinaryLabelsEncoder":
        assert isinstance(X, DataFrame)

        for col in self.columns:
            x = X[col]
            if issubdtype(x.dtype.type, integer):
                continue

            values = list(sorted(x.unique()))

            # skip if there are 2+ values
            if len(values) > 2:
                continue

            if len(values) == 1:
                v = values[0]
                map = {v: 0}
            else:
                u, v = values
                map = {u: 0, v: 1}

            self._maps[col] = map
        # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if self.copy: X = X.copy()

        for col in self.columns:
            if col not in self._maps:
                continue

            map = self._maps[col]
            X.replace({col: map}, inplace=True)
        # end
        return X
# end


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder(BaseEncoder):

    def __init__(self, columns, copy=True):
        super().__init__(columns, copy)
        self._maps = {}

    def fit(self, X: DataFrame) -> "OneHotEncoder":
        assert isinstance(X, DataFrame)

        for col in self.columns:
            values = sorted(set(X[col].to_list()))
            n = len(values)
            map = {}
            for i in range(n):
                v = values[i]
                map[v] = i

            self._maps[col] = map

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)
        if self.copy: X = X.copy()

        for col in self.columns:
            map = self._maps[col]

            for v in map:
                vcol = f"{col}_{v}"
                X[vcol] = 0
                X.loc[X[col] == v, vcol] = 1
            # end

        return X
# end
