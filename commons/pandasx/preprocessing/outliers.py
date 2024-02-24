from typing import Union
from pandas import Series, DataFrame

from .base import GroupsBaseEncoder


# ---------------------------------------------------------------------------
# OutlierTransformer
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class OutlierTransformer(GroupsBaseEncoder):
    """
    Compute mean and standard deviation, then replace the outliers based on some strategy.
    Supported strategies:

        median  replace with the median value
        mean    replace with the mean value
        clip    clip the values in the range (mean +- outlier_std*std)

    If is is specified 'sp', the outliers ar computed on a season based.
    """

    def __init__(self, columns, *,
                 outlier_std=3, sp=None, strategy='clip',
                 groups=None, copy=True):
        super().__init__(columns, groups, copy)
        self.sp = sp
        self.outlier_std = outlier_std
        self.strategy = strategy

        self._means = {}
        self._stds = {}
        self._medians = {}

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        if g is None:
            return self._means, self._stds, self._medians
        else:
            return self._means[g], self._stds[g], self._medians[g]

    def _set_params(self, g, params):
        means, stds, medians = params
        if g is None:
            self._means = means
            self._stds = stds
            self._medians = medians
        else:
            self._means[g] = means
            self._stds[g] = stds
            self._medians[g] = medians
        pass

    def _compute_params(self, g, X):
        if self.sp is None:
            return self._compute_means_stds(X)
        else:
            return self._compute_seasonal_means_stds(X)

    def _apply_transform(self, X, params):
        means, stds, medians = params
        if self.sp is None:
            return self._transform(X, means, stds, medians)
        else:
            return self._seasonal_transform(X, means, stds, medians)

    # -----------------------------------------------------------------------

    def _compute_means_stds(self, X: DataFrame) -> tuple[dict, dict, dict]:
        means = {}
        stds = {}
        medians = {}
        for col in self._get_columns(X):
            x = X[col]

            means[col] = x.mean()
            stds[col] = x.std()
            medians[col] = x.median()
        return means, stds, medians
    # end

    def _compute_seasonal_means_stds(self, X: DataFrame) -> tuple[dict, dict, dict]:
        sp = self.sp

        means = {}
        stds = {}
        medians = {}
        for col in self._get_columns(X):
            x = X[col]

            n = len(x)
            s = n % sp

            means_s = []
            stds_s = []
            medians_s = []

            for i in range(s, n, sp):
                if i == s:
                    xs = x[0:i+sp]
                else:
                    xs = x[i:i + sp]

                means_s.append(xs.mean())
                stds_s.append(xs.std())
                medians_s.append(xs.median())
            # end

            means[col] = means_s
            stds[col] = stds_s
            medians[col] = medians_s
        return means, stds, medians
    # end

    # -----------------------------------------------------------------------

    def _transform(self, X: DataFrame, means, stds, medians) -> DataFrame:
        X = self._check_X(X)
        ostd = self.outlier_std

        if ostd in [None, 0]:
            return X

        columns = self._get_columns(X)
        for col in columns:
            x: Series = X[col].copy()

            mean = means[col]
            std = stds[col]
            median = medians[col]

            min_value = mean - ostd*std
            max_value = mean + ostd*std
            mean_value = mean
            median_value = median

            if min_value <= x.min() <= x.max() <= max_value:
                pass
            elif self.strategy == 'median':
                x[(x <= min_value) | (x >= max_value)] = median_value
            elif self.strategy == 'mean':
                x[(x <= min_value) | (x >= max_value)] = mean_value
            elif self.strategy == 'clip':
                x[(x <= min_value)] = min_value
                x[(x >= max_value)] = max_value
            else:
                raise ValueError(f'Unsupported strategy {self.strategy}')
            X[col] = x
        return X
    # end

    def _seasonal_transform(self, X: DataFrame, means, stds, medians) -> DataFrame:
        X = self._check_X(X)
        sp = self.sp
        ostd = self.outlier_std

        if ostd in [None, 0]:
            return X

        columns = self._get_columns(X)
        for col in columns:
            x_: Series = X[col].copy()

            means_s = means[col]
            stds_s = stds[col]
            medians_s = medians[col]

            n = len(x_)
            s = n % sp

            for i in range(s, n, sp):
                if i == s:
                    j = 0
                    x = x_[0:i+sp]
                else:
                    j = (i-s)//sp
                    x = x_[i:i+sp]

                mean = means_s[j]
                std = stds_s[j]
                median = medians_s[j]

                min_value = mean - ostd*std
                max_value = mean + ostd*std
                mean_value = mean
                median_value = median

                if min_value <= x.min() <= x.max() <= max_value:
                    pass
                elif self.strategy == 'median':
                    x[(x <= min_value) | (x >= max_value)] = median_value
                elif self.strategy == 'mean':
                    x[(x <= min_value) | (x >= max_value)] = mean_value
                elif self.strategy == 'clip':
                    x[(x <= min_value)] = min_value
                    x[(x >= max_value)] = max_value
                else:
                    raise ValueError(f'Unsupported strategy {self.strategy}')

                if i == s:
                    x_[0:i+sp] = x
                else:
                    x_[i:i+sp] = x
            # end
            X[col] = x_
        return X
# end


# ---------------------------------------------------------------------------
# QuantileTransformer
# ---------------------------------------------------------------------------

class QuantileTransformer(GroupsBaseEncoder):

    def __init__(self, columns: Union[None, str, list[str]] = None,
                 quantile=.05,
                 *,
                 groups: Union[None, str, list[str]] = None,
                 copy=True):
        """
        Apply the scaler to the selected columns.
        If no column is specified, the scaler is applied to all columns
        If it is specified 'groups' or the has a MultIndex, the scaling is applied on
            'per-group' basis

        :param columns: column or columns where to apply the scaling.
                If None, the scaling is applied to all columns
        :param quantile: quantile to use. It can be specified as:
                n:int   the min/max values are computed excluding n values
                        from the minimum and maximum
                f:float the min/max values are computed excluding (n*f) values
                        from the minimum and maximum
        :param groups: columns used to identify the TS in a multi-TS dataset
                If None, it is used the MultiIndex
        """
        super().__init__(columns, groups, copy)
        self.quantile = quantile
        # self.groups = as_list(groups, 'groups')

        self._mins = {}
        self._maxs = {}

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        if g is None:
            return self._mins, self._maxs
        else:
            return self._mins[g], self._maxs[g]

    def _set_params(self, g, params):
        mins, maxs = params
        if g is None:
            self._mins = mins
            self._maxs = maxs
        else:
            self._mins[g] = mins
            self._maxs[g] = maxs
        pass

    def _compute_params(self, g, X):
        return self._compute_mins_maxs(X)

    def _apply_transform(self, X, params):
        mins, maxs = params
        return self._transform(X, mins, maxs)

    # -----------------------------------------------------------------------

    def _compute_mins_maxs(self, X):
        n = len(X)
        nskip = self.quantile if self.quantile >= 1 else max(1, int(self.quantile*n))
        assert 2*nskip < n

        mins = {}
        maxs = {}
        columns = self._get_columns(X)
        for col in columns:
            x = X[col].to_numpy()
            x.sort()

            minv = x[nskip]
            maxv = x[n-nskip-1]

            mins[col] = minv
            maxs[col] = maxv
        # end
        return mins, maxs

    def _transform(self, X, mins, maxs):

        columns = self._get_columns(X)
        for col in columns:
            minv = mins[col]
            maxv = maxs[col]

            x = X[col]
            x[x < minv] = minv
            x[x > maxv] = maxv

            X[col] = x
        return X
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
