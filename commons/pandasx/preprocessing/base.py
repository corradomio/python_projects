from typing import Union

import numpy as np
from pandas import DataFrame, Series

from stdlib import NoneType, as_list


class BaseEncoder:

    def __init__(self, columns, copy):
        self.columns = as_list(columns)
        self.copy = copy
        self._col = self.columns[0] if len(self.columns) > 0 else None
        pass

    def fit(self, X) -> "BaseEncoder":
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X) -> DataFrame:
        ...

    def fit_transform(self, X) -> DataFrame:
        return self.fit(X).transform(X)

    def _get_columns(self, X) -> list[str]:
        if len(self.columns) > 0:
            return self.columns
        else:
            return list(X.columns)

    def _check_X(self, X):
        assert isinstance(X, DataFrame)
        if self.copy:
            return X.copy()
        else:
            return X

# end


class XyBaseEncoder:

    def __init__(self, columns, copy):
        self.columns = as_list(columns)
        self.copy = copy
        self._col = self.columns[0] if len(self.columns) > 0 else None
        pass

    def fit(self, X, y) -> "XyBaseEncoder":
        assert isinstance(X, (NoneType, DataFrame, Series))
        assert isinstance(y, (DataFrame, Series))
        return self

    def transform(self, X, y) -> Union[DataFrame, tuple[DataFrame, DataFrame], tuple[np.ndarray, np.ndarray]]:
        ...

    def fit_transform(self, X, y) -> DataFrame:
        return self.fit(X, y).transform(X, y)

    def _get_columns(self, X) -> list[str]:
        if len(self.columns) > 0:
            return self.columns
        else:
            return list(X.columns)

    def _check_Xy(self, X, y):
        assert isinstance(X, (DataFrame, Series))
        assert isinstance(y, (DataFrame, Series))
        return X, y

    def _check_X(self, X):
        assert isinstance(X, (DataFrame, Series))
        return X

# end
