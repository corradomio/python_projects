import pandas as pd

from .base import PandasBaseEncoder
from ..base import PANDAS_TYPE


# ---------------------------------------------------------------------------
# IdentityScaler
# ---------------------------------------------------------------------------

class IdentityScaler(PandasBaseEncoder):
    def __init__(self, columns=None,copy=True):
        super().__init__(columns, copy)

    def fit(self, X):
        return self

    def transform(self, X: PANDAS_TYPE) -> PANDAS_TYPE:
        return X

    def inverse_transform(self, X: PANDAS_TYPE) -> PANDAS_TYPE:
        return X
# end


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler(PandasBaseEncoder):

    def __init__(self, columns=None,
                 feature_range=(0, 1),
                 *,
                 outlier_std=0.,
                 method=None,
                 clip=None,
                 copy=True):
        super().__init__(columns, copy)
        self.feature_range = feature_range
        self.outlier_std = outlier_std
        self.clip = clip
        self.method = method

        self._meanf = float(feature_range[0])
        self._sdevf = float(feature_range[1])
        self._means = {}
        self._sdevs = {}
    # end

    def _fit(self, S: pd.Series):
        name = S.name
        mean = S.median() if self.method == "median" else S.mean()
        stdv = S.std()
        self._means[name] = mean
        self._sdevs[name] = stdv
    # end

    def _transform(self, S: pd.Series) -> pd.Series:
        name = S.name
        if name not in self._means:
            return S

        mean = self._means[name]
        stdv = self._sdevs[name]
        T = (S-mean)/stdv

        meanf = self._meanf
        stdvf = self._sdevf
        T = (T*stdvf) + meanf
        return T
    # end

    def _inverse_transform(self, S: pd.Series) -> pd.Series:
        name = S.name
        if name not in self._meanf:
            return S

        meanf = self._meanf
        stdvf = self._sdevf
        I = (S-meanf)/stdvf

        mean = self._means[name]
        stdv = self._sdevs[name]
        I = (I*stdv) + mean
        clip = self.clip
        if clip is not None:
            I[I<clip] = clip
        return I
    # end
# end

StandardScalerEncoder = StandardScaler


# ---------------------------------------------------------------------------
# LinearMinMaxScaler
# ---------------------------------------------------------------------------

EPS = 1.0e-6

class MinMaxScaler(PandasBaseEncoder):

    def __init__(self, columns=None,
                 feature_range=(-1, 1),
                 *,
                 outlier_std=0,
                 clip=None,
                 copy=True):
        """
        :param columns: columns to process. If None, all columns
        :param feature_range: (min, max) ranges
        :param outlier_std: outlier values. If > 0, values outside [mean-ostd*std,mean+ostd*std]
                are clipped
        :param clip: extra clip. If not None, values < clip are replaces with 'clip'
                Used in inverse_transform
        :param copy
        """
        super().__init__(columns, copy)
        self.feature_range = feature_range
        self.outlier_std = outlier_std
        self.clip = clip

        self._minf = float(feature_range[0])
        self._maxf = float(feature_range[1])
        self._mins = {}
        self._maxs = {}
    # end

    def _fit(self, S: pd.Series):
        name = S.name
        minv = S.min()
        maxv = S.max()

        # skip encoding if already in the range
        if self._minf-EPS <= minv and maxv <= self._maxf+EPS:
            return

        if self.outlier_std > 0:
            mean = S.mean()
            stdv = S.std()
            minf = mean - self.outlier_std*stdv
            if minv < minf:
                minv = minf
            maxf = mean + self.outlier_std*stdv
            if maxv > maxf:
                maxv = maxf
        # end

        self._mins[name] = minv
        self._maxs[name] = maxv
    # end

    def _transform(self, S: pd.Series) -> pd.Series:
        name = S.name
        if name not in self._mins:
            return S

        minv = self._mins[name]
        maxv = self._maxs[name]
        T = (S-minv)/(maxv-minv)

        minf = self._minf
        maxf = self._maxf
        T = (T*(maxf-minf)) + minf
        return T
    # end

    def _inverse_transform(self, S: pd.Series) -> pd.Series:
        name = S.name
        if name not in self._mins:
            return S

        minf = self._minf
        maxf = self._maxf
        I = (S-minf)/(maxf-minf)

        minv = self._mins[name]
        maxv = self._maxs[name]
        I = (I*(maxv-minv)) + minv

        clip = self.clip
        if clip is not None:
            I[I<clip] = clip
        return I
    # end
# end

MinMaxScalerEncoder = MinMaxScaler

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
