from typing import Union
from pandas import Series, DataFrame

from .base import BaseEncoder


# ---------------------------------------------------------------------------
# OutlierTransformer
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class OutlierTransformer(BaseEncoder):
    """
    Compute mean and standard deviation, then replace the outliers based on some strategy.
    Supported strategies:

        median  replace with the median value
        mean    replace with the mean value
        clip    clip the values in the range (mean +- outlier_std*std)

    If is is specified 'sp', the outliers ar computed on a season based.
    """

    def __init__(self, columns=None, *,
                 outlier_std=3,
                 mode='clip',
                 copy=True):
        super().__init__(columns, copy)
        self.outlier_std = outlier_std
        self.mode = mode

        assert mode in ["median", "mean", "clip"], f"Unsupported mode {mode}"

        self._means = {}
        self._stds = {}
        self._medians = {}

    # -----------------------------------------------------------------------

    def fit(self, X: DataFrame) -> "OutlierTransformer":
        for col in self._get_columns(X):
            if col not in X.columns:
                continue
            self._means[col] =  X[col].mean()
            self._stds[col] = X[col].std()
            self._medians[col] = X[col].median()
        # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        outlier_std = self.outlier_std

        if outlier_std is None or outlier_std <= 0 or len(self._means) == 0:
            return X

        for col in self._means:
            if col not in X.columns:
                continue

            mean = self._means[col]
            std = self._stds[col]
            median = self._medians[col]
            vmin = mean - outlier_std * std
            vmax = mean + outlier_std * std

            if std < NO_SCALE_EPS:
                std = NO_SCALE_LIMIT

            if self.mode == 'clip':
                X.loc[X[col] < vmin, col] = vmin
                X.loc[X[col] > vmax, col] = vmax
            elif self.mode == 'median':
                X.loc[X[col] < vmin, col] = median
                X.loc[X[col] > vmax, col] = median
            elif self.mode == 'mean':
                X.loc[X[col] < vmin, col] = mean
                X.loc[X[col] > vmax, col] = mean
            else:
                raise ValueError(f'Unsupported mode {self.mode}')

        return X
# end
