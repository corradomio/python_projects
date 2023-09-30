from pandas import DataFrame

from stdlib import as_list


class BaseEncoder:

    def __init__(self, columns, copy):
        self.columns = as_list(columns)
        self.copy = copy
        self._col = self.columns[0] if len(self.columns) > 0 else None
        pass

    def fit(self, X: DataFrame) -> "BaseEncoder":
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        ...

    def fit_transform(self, X: DataFrame) -> DataFrame:
        return self.fit(X).transform(X)

    def _get_columns(self, X: DataFrame) -> list[str]:
        if len(self.columns) > 0:
            return self.columns
        else:
            return list(X.columns)

    def _check_X(self, X: DataFrame):
        assert isinstance(X, DataFrame)
        if self.copy:
            return X.copy()
        else:
            return X

# end
