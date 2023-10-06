from pandas import DataFrame, Series

from stdlib import NoneType, as_list


class BaseEncoder:

    def __init__(self, columns, copy):
        self.columns = as_list(columns, "columns")
        self.copy = copy
        self._col = self.columns[0] if len(self.columns) > 0 else None
        pass

    def fit(self, X, y=None) -> "BaseEncoder":
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X) -> DataFrame:
        ...

    def fit_transform(self, X, y=None) -> DataFrame:
        return self.fit(X, y).transform(X)

    def _get_columns(self, X) -> list[str]:
        if len(self.columns) > 0:
            return self.columns
        else:
            return list(X.columns)

    def _check_X(self, X, y=None):
        assert isinstance(X, DataFrame)
        assert y is None, "Parameter y must e None"
        return X.copy() if self.copy else X

    def _check_Xy(self, X, y):
        assert isinstance(X, DataFrame)
        assert isinstance(y, (NoneType, DataFrame))
        return X.copy() if self.copy else X, y
# end


class XyBaseEncoder(BaseEncoder):

    def __init__(self, columns, copy):
        super().__init__(columns, copy)

    def fit(self, X, y=None) -> "XyBaseEncoder":
        assert isinstance(X, (NoneType, DataFrame, Series))
        assert isinstance(y, (NoneType, DataFrame, Series))
        return self

    def transform(self, X, y=None):
        assert isinstance(X, (NoneType, DataFrame, Series))
        assert isinstance(y, (NoneType, DataFrame, Series))
        return None

    def fit_transform(self, X, y=None):
        return self.fit(X=X, y=y).transform(X=X, y=y)

# end
