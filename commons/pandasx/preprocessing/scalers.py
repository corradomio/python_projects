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

            if 0 <= min(x) <= max(x) <= 1:
                continue

            means[col] = x.mean()
            sdevs[col] = x.std()
        # end
        return means, sdevs

    def _transform(self, X: pd.DataFrame, means, sdevs) -> pd.DataFrame:
        X = X.copy()
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

    def _scale(self, X: pd.DataFrame, means, sdevs):
        meanv = self._meanv
        sdevv = self._sdevv

        for col in self._get_columns(X):
            if col not in means:
                continue

            meanc = means[col]
            sdevc = sdevs[col]

            x = X[col].to_numpy(dtype=float)

            if sdevc > 0:
                x = meanv + (x - meanc) / sdevc * sdevv
            else:   # sdevc is 0 for CONSTANT values
                x = meanv

            X[col] = x
        return X

    def _inverse_transform(self, X, means, sdevs):
        X = X.copy()
        columns = self._get_columns(X)
        meanv = self._meanv
        sdevv = self._sdevv

        for col in columns:
            if col not in means:
                continue

            x = X[col].to_numpy(dtype=float)
            meanc = means[col]
            sdevc = sdevs[col]

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
# End
# ---------------------------------------------------------------------------
