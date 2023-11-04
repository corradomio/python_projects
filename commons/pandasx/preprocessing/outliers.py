from typing import Union

from pandas import Series, DataFrame

from .base import GroupsEncoder

# ---------------------------------------------------------------------------
# OutlierTransformer
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class OutlierTransformer(GroupsEncoder):
    """
    Compute mean and standard deviation, then replace the outliers based on some strategy.
    Supported strategies:

        median  replace with the median value
        mean    replace with the mean value
        min     replace with (mean - outlier_std*std)
        max     replace with (mean + outlier_std*std)
        clip    clip the values in the range (mean +- outlier_std*std)

    """

    def __init__(self, columns, outlier_std=4, strategy='clip', *, groups=None, copy=True):
        super().__init__(columns, groups, copy)
        self.outlier_std = outlier_std
        self.strategy = strategy

        self._means = {}
        self._sdevs = {}
        self._medians = {}

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        if g is None:
            return self._means, self._sdevs, self._medians
        else:
            return self._means[g], self._sdevs[g], self._medians[g]

    def _set_params(self, g, params):
        means, sdevs, medians = params
        if g is None:
            self._means = means
            self._sdevs = sdevs
            self._medians = medians
        else:
            self._means[g] = means
            self._sdevs[g] = sdevs
            self._medians[g] = medians
        pass

    def _compute_params(self, X):
        return self._compute_means_sdevs(X)

    def _apply_transform(self, X, params):
        means, sdevs, medians = params
        return self._transform(X, means, sdevs, medians)

    # -----------------------------------------------------------------------

    def _compute_means_sdevs(self, X: DataFrame):
        means = {}
        sdevs = {}
        medians = {}
        for col in self._get_columns(X):
            x = X[col]

            means[col] = x.mean()
            sdevs[col] = x.std()
            medians[col] = x.median()

            # x = X[col].to_numpy(dtype=float)
            # vmin, vmax = min(x), max(x)
            #
            # # if the values are already in a reasonable small range, don't scale
            # if -NO_SCALE_LIMIT <= vmin <= vmax <= +NO_SCALE_LIMIT:
            #     continue
            #
            # if (vmax - vmin) <= NO_SCALE_EPS:
            #     means[col] = x.mean()
            #     sdevs[col] = 0.
            # else:
            #     means[col] = x.mean()
            #     sdevs[col] = x.std()
            # # end
        return means, sdevs, medians

    # -----------------------------------------------------------------------

    def _transform(self, X: DataFrame, means, sdevs, medians) -> DataFrame:
        X = self._check_X(X)

        if self.outlier_std in [None, 0]:
            return X

        columns = self._get_columns(X)
        for col in columns:
            x: Series = X[col].copy()

            mean = means[col]
            sdev = sdevs[col]
            median = medians[col]

            min_value = mean - sdev
            max_value = mean + sdev
            mean_value = mean
            median_value = median

            if self.strategy == 'median':
                x[(x <= min_value) | (x >= max_value)] = median_value
            elif self.strategy == 'mean':
                x[(x <= min_value) | (x >= max_value)] = mean_value
            elif self.strategy == 'min':
                x[(x <= min_value) | (x >= max_value)] = min_value
            elif self.strategy == 'max':
                x[(x <= min_value) | (x >= max_value)] = max_value
            elif self.strategy == 'clip':
                x[(x <= min_value)] = min_value
                x[(x >= max_value)] = max_value
            else:
                raise ValueError(f'Unsupported strategy {self.strategy}')
            X[col] = x
        return X
    # end
# end


# ---------------------------------------------------------------------------
# QuantileTransformer
# ---------------------------------------------------------------------------

class QuantileTransformer(GroupsEncoder):

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
        :param groups: if the dataset contains multiple groups, column(s) used to identify each group
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

    def _compute_params(self, X):
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
