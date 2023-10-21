from typing import Union

import pandas as pd

from .base import GroupsEncoder

# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

NO_SCALE_EPS = 1.e-6


class StandardScaler(GroupsEncoder):

    def __init__(self, columns=None,
                 feature_range=(0, 1),
                 *,
                 outlier_std=0, clip=False,
                 groups=None, copy=True):
        """
        Apply the scaler to the selected columns.

        If no column is specified, the scaler is applied to all columns
        If it is specified 'groups' or it has a MultIndex, the scaling is applied on
            'per-group' basis

        :param columns: column or columns where to apply the scaling.
            If None, the scaling is applied to all columns
        :param feature_range: tuple (mean, std) values to use
        :param groups: if the dataset contains groups, column(s) used to identify each group
        """
        super().__init__(columns, groups, copy)
        self.feature_range = feature_range
        self.outlier_std = outlier_std
        self.clip = clip and outlier_std > 0

        self._meanv = float(feature_range[0])
        self._sdevv = float(feature_range[1])
        self._means = {}
        self._sdevs = {}

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        if g is None:
            return self._means, self._sdevs
        else:
            return self._means[g], self._sdevs[g]

    def _set_params(self, g, params):
        means, sdevs = params
        if g is None:
            self._means = means
            self._sdevs = sdevs
        else:
            self._means[g] = means
            self._sdevs[g] = sdevs
        pass

    def _compute_params(self, X):
        return self._compute_means_sdevs(X)

    def _apply_transform(self, X, params):
        means, sdevs = params
        return self._transform(X, means, sdevs)

    def _apply_inverse_transform(self, X, params):
        means, sdevs = params
        return self._inverse_transform(X, means, sdevs)

    # -----------------------------------------------------------------------

    def _compute_means_sdevs(self, X: pd.DataFrame):
        columns = self._get_columns(X)
        means = {}
        sdevs = {}
        for col in columns:

            x = X[col].to_numpy(dtype=float)
            vmin, vmax = min(x), max(x)

            if (vmax - vmin) <= NO_SCALE_EPS:
                means[col] = x.mean()
                sdevs[col] = 0.
            else:
                means[col] = x.mean()
                sdevs[col] = x.std()
            # end
        return means, sdevs

    def _transform(self, X: pd.DataFrame, means, sdevs) -> pd.DataFrame:
        X = self._clip(X, means, sdevs)
        X = self._scale(X, means, sdevs)
        return X

    def _clip(self, X, means, sdevs):
        if not self.clip:
            return X

        outlier_std = self.outlier_std
        for col in self._get_columns(X):
            if col not in means:
                continue

            meanc = means[col]
            sdevc = sdevs[col]
            minc = meanc - outlier_std*sdevc
            maxc = meanc + outlier_std*sdevc

            x = X[col].to_numpy(dtype=float)

            x[x < minc] = minc
            x[x > maxc] = maxc

            X[col] = x
        return X

    def _scale(self, X, means, sdevs):
        meanv = self._meanv
        sdevv = self._sdevv

        for col in self._get_columns(X):
            if col not in means:
                continue

            meanc = means[col]
            sdevc = sdevs[col]

            x = X[col].to_numpy(dtype=float)

            if sdevc <= NO_SCALE_EPS:
                x = meanv + (x - meanc) * sdevv
            else:
                x = meanv + (x - meanc) / sdevc * sdevv

            X[col] = x
        return X

    def _inverse_transform(self, X, means, sdevs):
        columns = self._get_columns(X)
        meanv = self._meanv
        sdevv = self._sdevv

        for col in columns:
            x = X[col].to_numpy(dtype=float)
            meanc = means[col]
            sdevc = sdevs[col]

            if sdevc <= NO_SCALE_EPS:
                x = (x - meanv)/sdevv + meanc
            else:
                x = (x - meanv)/sdevv*sdevc + meanc

            X[col] = x
        return X

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# compatibility
StandardScalerEncoder = StandardScaler


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler(GroupsEncoder):

    def __init__(self, columns: Union[None, str, list[str]] = None,
                 feature_range=(0, 1),
                 *,
                 quantile=.05, clip=False,
                 groups=None, copy=True):
        """
        Apply the scaler to the selected columns.
        If no column is specified, the scaler is applied to all columns
        If it is specified 'groups' or the has a MultIndex, the scaling is applied on
            'per-group' basis

        :param columns: column or columns where to apply the scaling.
            If None, the scaling is applied to all columns
        :param feature_range: tuple (min, max) values to use
        :param quantile: used to exclude too low or too high values
            Can be an integer value or a value < 1.
        :param clip: if to clip the outlier values
        :param groups: if the dataset contains multiple groups, column(s) used to identify each group
        """
        super().__init__(columns, groups, copy)
        self.feature_range = feature_range
        self.quantile = quantile
        self.clip = clip and quantile > 0

        self._minv = float(feature_range[0]),
        self._deltav = float(feature_range[1] - feature_range[0])

        self._mins = {}
        self._deltas = {}

        assert self._deltav > 0, f"Invalid feature_range: min must be <= max: {feature_range}"

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        if g is None:
            return self._mins, self._deltas
        else:
            return self._mins[g], self._deltas[g]

    def _set_params(self, g, params):
        mins, deltas = params
        if g is None:
            self._mins = mins
            self._deltas = deltas
        else:
            self._mins[g] = mins
            self._deltas[g] = deltas
        pass

    def _compute_params(self, X):
        return self._compute_mins_deltas(X)

    def _apply_transform(self, X, params):
        mins, deltas = params
        return self._transform(X, mins, deltas)

    def _apply_inverse_transform(self, X, params):
        mins, deltas = params
        return self._inverse_transform(X,mins, deltas)

    # -----------------------------------------------------------------------

    def _compute_mins_deltas(self, X):
        mins = {}
        deltas = {}

        n = len(X)
        nskip = self.quantile if self.quantile >= 1 else int(self.quantile*n)

        for col in self._get_columns(X):
            x = X[col].to_numpy()
            x.sort()

            minc = x[nskip]
            maxc = x[n - nskip - 1]

            mins[col] = minc
            deltas[col] = maxc - minc if maxc > (minc + NO_SCALE_EPS) else 1.
        # end
        return mins, deltas

    def _transform(self, X, mins, deltas):
        X = self._clip(X, mins, deltas)
        X = self._scale(X, mins, deltas)
        return X

    def _clip(self, X, mins, deltas):
        if not self.clip:
            return X

        for col in self._get_columns(X):
            minc = mins[col]
            deltac = deltas[col]
            maxc = minc + deltac

            x = X[col]
            x[x < minc] = minc
            x[x > maxc] = maxc

            X[col] = x
        return X

    def _scale(self, X, mins, deltas):
        minv = self._minv
        deltav = self._deltav

        columns = self._get_columns(X)
        for col in columns:
            minc = mins[col]
            deltac = deltas[col]

            x = X[col]
            x = minv + deltav * (x - minc) / deltac

            X[col] = x
        return X

    def _inverse_transform(self, X, mins, deltas):
        minv = self._minv
        deltav = self._deltav

        for col in self._get_columns(X):
            minc = mins[col]
            deltac = deltas[col]

            values = (X[col].to_numpy(dtype=float) - minv)/deltav
            values = minc + deltac*values

            X[col] = values
        return X
# end


# compatibility
MinMaxEncoder = MinMaxScaler

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
