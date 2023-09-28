from .base import *


# ---------------------------------------------------------------------------
# DTypeEncoder
# ---------------------------------------------------------------------------

class DTypeEncoder(BaseEncoder):
    # convert the selected columns to the specified dtype

    def __init__(self, columns, dtype, copy=True):
        super().__init__(columns, copy)
        self.dtype = dtype

    def transform(self, X: DataFrame) -> DataFrame:
        if self.copy: X = X.copy()

        for col in self.columns:
            X[col] = X[col].astype(self.dtype)
        return X
# end


# ---------------------------------------------------------------------------
# OrderedLabelEncoder
# ---------------------------------------------------------------------------

class OrderedLabelEncoder(BaseEncoder):
    # As an One Hot encoding but it ensure that the labels have a unique order

    def __init__(self, columns, mapping=(), remove_chars=None, copy=True):
        super().__init__(columns, copy)
        assert isinstance(mapping, (list, tuple))

        self.mapping: list[str] = mapping
        self.remove_chars = '()[]{}' if remove_chars is None else remove_chars

    def fit(self, X: DataFrame):
        assert isinstance(X, DataFrame)
        if len(self.mapping) == 0:
            col = self.columns[0]
            self.mapping = sorted(X[col].unique())
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if self.copy: X = X.copy()
        for col in self.columns:
            if len(self.mapping) <= 2:
                X = self._map_single_column(X, col)
            else:
                X = self._map_multiple_colmns(X, col)
        return X

    def _map_single_column(self, X, col):
        l = self.mapping
        n = len(l)
        mapping = {l[i]: i for i in range(n)}
        X[col] = X[col].replace(mapping)
        return X

    def _map_multiple_colmns(self, X, col: str):

        def ccol_of(key: str):
            if not isinstance(key, str):
                key = str(key)
            if ' ' in key:
                key = key.replace(' ', '_')
            for c in self.remove_chars:
                if c in key:
                    key = key.replace(c, '')
            return col + "_" + key

        for key in self.mapping:
            ccol = ccol_of(key)
            X[ccol] = 0
            X.loc[X[col] == key, ccol] = 1

        X = X[X.columns.difference([col])]
        return X
# end


# ---------------------------------------------------------------------------
# PandasCategoricalEncoder
# ---------------------------------------------------------------------------

class PandasCategoricalEncoder(BaseEncoder):
    # Apply the pandas '.astype("category")' to the column

    def __init__(self, columns, copy=True):
        super().__init__(columns, copy)

    def transform(self, X: DataFrame) -> DataFrame:
        if self.copy: X = X.copy()

        for col in self.columns:
            X[col] = X[col].astype('category')
        return X
# end

