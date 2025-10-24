from typing import Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ..utils import is_instance, NoneType, to_numpy

ARRAY_OR_DF = Union[NoneType, np.ndarray, pd.DataFrame]


# ---------------------------------------------------------------------------
# TimeseriesTransform
# ---------------------------------------------------------------------------

class TimeseriesTransform:

    def __init__(self, xlags, ylags, tlags, ulags=[]):
        assert is_instance(xlags, list[int]), f"Invalid 'xlags' value: {xlags}"
        assert is_instance(ylags, list[int]), f"Invalid 'ylags' value: {ylags}"
        assert is_instance(tlags, list[int]), f"Invalid 'tlags' value: {tlags}"
        assert is_instance(ulags, list[int]), f"Invalid 'ulags' value: {ulags}"

        self.xlags: list[int] = xlags
        self.ylags: list[int] = ylags
        self.tlags: list[int] = tlags
        self.ulags: list[int] = ulags
        pass
    # end

    # ---------------------------------------------------------------------------
    # fit/transform
    # ---------------------------------------------------------------------------

    def fit(self, *, y, X=None) -> "TimeseriesTransform":
        return self

    def transform(self, *, fh=None, y=None, X=None) -> tuple:
        return X, y

    def fit_transform(self, *, y=None, X=None, fh=None):
        return self.fit(y=y, X=X).transform(y=y, X=X, fh=fh)

    # DEBUG
    # def fit_debug(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None, fh=None):
    #     Xt, yt = self.fit_transform(y=y, X=X, fh=fh)
    #     zt = np.zeros((len(yt), 1), dtype=y.dtype)
    #     return np.concat([Xt, zt, yt], axis=-1)

    # ---------------------------------------------------------------------------
    # Implementation
    # ---------------------------------------------------------------------------

    def _check_Xy(self, X, y=None, fh=None):
        X = to_numpy(X)
        y = to_numpy(y, matrix=True)
        assert fh is None
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, (NoneType, np.ndarray))
        return X, y

    def _check_Xfh(self, X, fh, y):
        assert y is None
        # Note: to be compatible with 'sktime' fh MUST starts 1 timeslot after the
        # 'cutoff', that is, the LAST timestamp used in training. This means that
        # If specified as list of as a ForecastingHorizon, it MUST be:
        #
        #   [1,2,3.....]
        #
        # Then, the LAST value is the 'prediction_length'.
        #
        # dtype = dtype_of(X, y)
        # X = to_numpy(X, dtype=dtype)
        X = to_numpy(X)

        if fh is None or fh <= 0:
            assert X is not None, "If fh is not specified, X must be not None"
            fh = len(X)
        elif isinstance(fh, list):
            assert len(fh) > 0 and fh[0] >= 1, f'fh can not start with a value < 1 with ({fh[0]})'
            fh = fh[-1]
        elif isinstance(fh, ForecastingHorizon):
            assert fh.is_relative, f'fh must be a relative ForecastingHorizon'
            assert len(fh) > 0 and fh[0] >= 1, f'fh can not start with a value < 1 with ({fh[0]})'
            fh = fh[-1]

        assert isinstance(fh, (int, np.int64))

        # check if X has valid type and fh < |X|
        if X is not None:
            assert isinstance(X, np.ndarray)
            assert fh <= len(X), f'fh larger than |X|'

        return X, fh
# end


# ---------------------------------------------------------------------------
#   End
# ---------------------------------------------------------------------------
