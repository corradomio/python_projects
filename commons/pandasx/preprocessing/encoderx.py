from typing import Optional

from pandas import DataFrame
from .base import BaseEncoder


# ---------------------------------------------------------------------------
# DTypeEncoder
# ---------------------------------------------------------------------------

class DTypeEncoder(BaseEncoder):
    """
    Apply '.astype(T)' to the selected columns
    """

    def __init__(self, columns=None, dtype=float, copy=True):
        super().__init__(columns, copy)
        self.dtype = dtype

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        for col in self._get_columns(X):
            X[col] = X[col].astype(self.dtype)
        return X
# end


# ---------------------------------------------------------------------------
# OrderedLabelEncoder
# ---------------------------------------------------------------------------

class OrderedLabelsEncoder(BaseEncoder):
    # As an One Hot encoding but it ensure that the labels have a unique order
    # It is NOT reasonable to split the dataset in groups because it is possible
    # that each group can have different number of labels, than the number of
    # onehot columns can be different. This has the following effect: it is not
    # possible to use the same model (or its clone) to each group!
    #

    def __init__(self, columns=None, mapping=None, remove_chars=None, copy=True):
        super().__init__(columns, copy)
        self.mapping: Optional[list[str]] = mapping
        self.remove_chars = '()[]{}' if remove_chars is None else remove_chars

        self._mapping = {}

    def fit(self, X: DataFrame):
        X = self._check_X(X)

        for col in self._get_columns(X):
            unique = sorted(X[col].unique())
            self._mapping[col] = {unique[i]: i for i in range(len(unique))}
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        for col in self._get_columns(X):
            mapping = self._get_mapping(col)

            if len(mapping) <= 2:
                X = self._map_single_column(X, col)
            else:
                X = self._map_multiple_columns(X, col)
        return X

    def _get_mapping(self, col):
        return self.mapping if self.mapping else self._mapping[col]

    def _map_single_column(self, X, col):
        mapping = self._get_mapping(col)
        X[col] = X[col].replace(mapping)
        return X

    def _map_multiple_columns(self, X, col: str):
        mapping = self._get_mapping(col)

        def ccol_of(key: str):
            if not isinstance(key, str):
                key = str(key)
            if ' ' in key:
                key = key.replace(' ', '_')
            for c in self.remove_chars:
                key = key.replace(c, '')
                # if c in key:
                #     key = key.replace(c, '')
            return f"{col}_{key}"

        for key in mapping:
            ccol = ccol_of(key)
            X[ccol] = 0
            X.loc[X[col] == key, ccol] = 1

        # remove the original column
        X.drop([col], axis=1, inplace=True)
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
        X = self._check_X(X)

        for col in self._get_columns(X):
            X[col] = X[col].astype('category')
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
