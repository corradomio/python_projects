import pandas as pd
from .base import BaseEncoder, as_list
from ..base import groups_split, groups_merge

# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class StandardScaler(BaseEncoder):
    # (mean, sdev) scaler for each column

    def __init__(self, columns=None, groups=None, copy=True):
        super().__init__(columns, copy)
        self.groups = as_list(groups, "groups")
        self._means = {}

    def fit(self, X: pd.DataFrame, y=None):
        self._check_X(X, y)

        if len(self.groups) == 0:
            self._means = self._compute_means(X)
            return self

        groups = groups_split(X, groups=self.groups, drop=True)
        for g in groups:
            Xg = groups[g]
            means = self._compute_means(Xg)
            self._means[g] = means

        return self

    def _compute_means(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)

        columns = self._get_columns(X)
        mean_stdv = {}
        for col in columns:

            x = X[col].to_numpy(dtype=float)
            vmin, vmax = min(x), max(x)

            # if the values are already in a reasonable small range, don't scale
            if -NO_SCALE_LIMIT <= vmin <= vmax <= +NO_SCALE_LIMIT:
                continue

            if (vmax - vmin) <= NO_SCALE_EPS:
                mean_stdv[col] = (x.mean(), 0.)
            else:
                mean_stdv[col] = (x.mean(), x.std())
            # end
        return mean_stdv

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check_X(X)

        if len(self.groups) == 0:
            return self._transform(X, self._means)

        X_dict: dict = dict()
        groups = groups_split(X, groups=self.groups, drop=True)

        for g in groups:
            Xg = groups[g]
            means = self._means[g]
            Xg = self._transform(Xg, means)
            X_dict[g] = Xg

        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _transform(self, X: pd.DataFrame, means) -> pd.DataFrame:
        columns = self._get_columns(X)
        for col in columns:
            if col not in means:
                continue

            x = X[col].to_numpy(dtype=float)
            mean, sdev = means[col]

            if sdev <= NO_SCALE_EPS:
                x = (x - mean)
            else:
                x = (x - mean) / sdev

            X[col] = x
        return X

    def inverse_transform(self, X: pd.DataFrame):
        X = self._check_X(X)

        if len(self.groups) == 0:
            return self._inverse_transform(X, self._means)

        X_dict: dict = dict()
        groups = groups_split(X, groups=self.groups, drop=True)

        for g in groups:
            Xg = groups[g]
            means = self._means[g]
            Xg = self._inverse_transform(Xg, means)
            X_dict[g] = Xg

        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _inverse_transform(self, X, means):
        columns = self._get_columns(X)
        for col in columns:
            x = X[col].to_numpy(dtype=float)
            mean, sdev = means[col]

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

    def fit(self, X: pd.DataFrame, y=None) -> "MinMaxScaler":
        self._check_X(X, y)

        columns = self._get_columns(X)
        for col in columns:
            values = X[col].to_numpy(dtype=float)
            minval = min(values)
            deltaval = max(values) - minval

            self._min[col] = minval
            self._delta[col] = deltaval

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

    def inverse_transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
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
# End
# ---------------------------------------------------------------------------
