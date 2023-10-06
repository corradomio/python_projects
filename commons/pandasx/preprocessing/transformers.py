from .base import *
from pandas import Series


# ---------------------------------------------------------------------------
# OutlierTransformer
# ---------------------------------------------------------------------------

class OutlierTransformer(BaseEncoder):
    # Replace the outliers with an alternative value

    def __init__(self, columns, outlier_std=4, strategy='median', copy=True):
        super().__init__(columns, copy)
        self.outlier_std = outlier_std
        self.strategy = strategy

        self._mean = {}
        self._sdev = {}
        self._median = {}

    def fit(self, X: DataFrame, y=None) -> "OutlierTransformer":
        self._check_X(X, y)

        columns = self._get_columns(X)
        for col in columns:
            x: Series = X[col]
            self._mean[col] = x.mean()
            self._sdev[col] = x.std()
            self._median[col] = x.median()

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        if self.outlier_std in [None, 0]:
            return X

        columns = self._get_columns(X)
        for col in columns:
            x: Series = X[col].copy()
            mean = self._mean[col]
            sdev = self._sdev[col]

            min_value = mean - sdev
            max_value = mean + sdev
            mean_value = mean
            median_value = self._median[col]

            if self.strategy == 'median':
                x[(x <= min_value) | (x >= max_value)] = median_value
            elif self.strategy == 'mean':
                x[(x <= min_value) | (x >= max_value)] = mean_value
            elif self.strategy == 'min':
                x[(x <= min_value) | (x >= max_value)] = min_value
            elif self.strategy == 'max':
                x[(x <= min_value) | (x >= max_value)] = max_value
            else:
                raise ValueError(f'Unsupported strategy {self.strategy}')
            X[col] = x
        return X
    # end
# end


# ---------------------------------------------------------------------------
# IgnoreTransformer
# ---------------------------------------------------------------------------

class IgnoreTransformer(BaseEncoder):
    # Remove the specified columns

    def __init__(self, columns, keep=None, copy=True):
        """

        :param columns: columns to remove of None
        :param keep: columns to keep (as alternative to columns)
        """
        super().__init__(columns, copy)
        self.keep = as_list(keep, "keep")

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        columns = self._get_columns(X)
        keep = self.keep
        if keep:
            columns = set(columns).difference(keep)

        X.drop(columns, axis=1, inplace=True)
        return X
# end
