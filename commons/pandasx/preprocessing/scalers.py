from .base import *


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class StandardScaler(BaseEncoder):
    # (mean, sdev) scaler for each column

    def __init__(self, columns=None, copy=True):
        super().__init__(columns, copy)
        self._mean = {}
        self._sdev = {}

    def fit(self, X: DataFrame) -> "StandardScalerEncoder":
        assert isinstance(X, DataFrame)

        columns = self._get_columns(X)
        for col in columns:

            x = X[col].to_numpy(dtype=float)
            vmin, vmax = min(x), max(x)

            # if the values are already in a reasonable small range, don't scale
            if -NO_SCALE_LIMIT <= vmin <= vmax <= +NO_SCALE_LIMIT:
                continue

            if (vmax - vmin) <= NO_SCALE_EPS:
                self._mean[col] = x.mean()
                self._sdev[col] = 0.
            else:
                self._mean[col] = x.mean()
                self._sdev[col] = x.std()
            # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        columns = self._get_columns(X)
        for col in columns:
            if col not in self._mean:
                continue

            x = X[col].to_numpy(dtype=float)
            mean = self._mean[col]
            sdev = self._sdev[col]

            if sdev <= NO_SCALE_EPS:
                x = (x - mean)
            else:
                x = (x - mean) / sdev

            X[col] = x
        return X

    def inverse_transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)

        if self.copy:
            X = X.copy()

        columns = self._get_columns(X)
        for col in columns:
            x = X[col].to_numpy(dtype=float)
            mean = self._mean[col]
            sdev = self._sdev[col]

            if sdev <= NO_SCALE_EPS:
                x = x + mean
            else:
                x = x*sdev + mean

            X[col] = x
        return X
# end


# compatibility
StandardScalerEncoder = StandardScaler


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler(BaseEncoder):
    # (min, max) scaler for each column

    def __init__(self, columns, feature_range=(0, 1), *, copy=True):
        super().__init__(columns, copy)
        self.feature_range = feature_range

        self._min_value = feature_range[0],
        self._delta_values = feature_range[1] - feature_range[0]
        self._min = {}
        self._delta = {}

    def fit(self, X: DataFrame) -> "MinMaxScaler":
        assert isinstance(X, DataFrame)

        columns = self._get_columns(X)
        for col in columns:
            values = X[col].to_numpy(dtype=float)
            minval = min(values)
            deltaval = max(values) - minval

            self._min[col] = minval
            self._delta[col] = deltaval

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        minv = self._min_value
        deltav = self._delta_values

        columns = self._get_columns(X)
        for col in columns:
            minval = self._min[col]
            deltaval = self._delta[col]

            values = X[col].to_numpy(dtype=float)
            values = minv + deltav*(values-minval)/deltaval

            X[col] = values
        return X

    def inverse_transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)
        if self._copy: X = X.copy()

        minv = self._min_value
        deltav = self._delta_values

        columns = self.columns
        for col in columns:
            minval = self._min[col]
            deltaval = self._delta[col]

            values = (X[col].to_numpy(dtype=float) - minv)/deltav
            values = minval + deltaval*values

            X[col] = values
        return X
# end


# compatibility
MinMaxEncoder = MinMaxScaler


# ---------------------------------------------------------------------------
# MeanStdScaler
# ---------------------------------------------------------------------------

# class MeanStdScaler(BaseEncoder):
#     # (mean, sted) scaler for the COMPLETE dataframe
#
#     def __init__(self, columns, copy=True):
#         super().__init__(columns, copy)
#
#         self._mean = 0.
#         self._sdev = 0.
#
#     def fit(self, X: DataFrame) -> "MeanStdScaler":
#         assert isinstance(X, DataFrame)
#         col = self._col
#
#         values = X[col].to_numpy(dtype=float)
#         self._mean = values.mean()
#         self._sdev = values.std()
#         return self
#
#     def transform(self, X: DataFrame) -> DataFrame:
#         assert isinstance(X, DataFrame)
#         if self.copy: X = X.copy()
#
#         col = self._col
#         values = X[col].to_numpy(dtype=float)
#         values = (values - self._mean) / self._sdev
#
#         X[col] = values
#         return X
#
#     def inverse_transform(self, X: DataFrame):
#         assert isinstance(X, DataFrame)
#         if self.copy: X = X.copy()
#
#         col = self._col
#         values = X[col].to_numpy(dtype=float)
#         values = values*self._sdev + self._mean
#
#         X[col] = values
#         return X
# # end
#
#
# MeanStdEncoder = MeanStdScaler


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
