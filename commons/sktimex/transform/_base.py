from typing import Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ._lags import tlags_start
from ..utils import NoneType, to_matrix

ARRAY_OR_DF = Union[NoneType, np.ndarray, pd.DataFrame]


# ---------------------------------------------------------------------------
#   ModelTransform
#       ModelTrainTransform
#       ModePredictTransform
# ---------------------------------------------------------------------------

class TimeseriesTransform:

    def _check_Xy(self, X, y=None, fh=None):
        X = to_matrix(X)
        y = to_matrix(y)
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
        fh_ = fh
        X = to_matrix(X)
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

        assert isinstance(fh, int)

        # check if X has valid type and fh < |X|
        if X is not None:
            assert isinstance(X, np.ndarray)
            assert fh <= len(X), f'fh larger than |X|'

        return X, fh

    def fit(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None) -> "TimeseriesTransform":
        return self

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF = None, fh=None) -> tuple:
        return X, y

    def fit_transform(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None, fh=None):
        return self.fit(y=y, X=X).transform(y=y, X=X)
# end


class ModelTrainTransform(TimeseriesTransform):

    def __init__(self, xlags, ylags, tlags):
        self.xlags: list = xlags
        self.ylags: list = ylags
        self.tlags: list = tlags
    # end
# end


class ModelPredictTransform(TimeseriesTransform):

    def __init__(self, xlags, ylags, tlags):
        self.xlags: list = xlags
        self.ylags: list = ylags
        self.tlags: list = tlags
        self.tstart: int = tlags_start(tlags)

        self.Xh = None  # X history
        self.yh = None  # y history

        self.Xp = None  # X prediction past
        self.yp = None  # y prediction future
    # end

    def fit(self, y: ARRAY_OR_DF, X: ARRAY_OR_DF = None):
        # used to support the implementation of 'self.to_pandas()'
        self.y_pandas = y

        # This method must be used to pass 'y_train' and 'X_train'
        X, y = self._check_Xy(X, y)

        self.Xh = X
        self.yh = y

        return self
    # end

    def transform(self, fh: int, X: ARRAY_OR_DF = None, y=None):
        assert isinstance(fh, int)
        assert isinstance(X, ARRAY_OR_DF)

        # This method is used with 'fh' and 'X_test'/'X_predict'
        X, fh = self._check_Xfh(X, fh, y)

        self.Xp = X
        self.fh = fh

        return fh, X
    # end

    def update(self, i: int, y_pred, t=None):
        # Note:
        #   the parameter 't' is used to override tlags
        #   'tlags' is at minimum [0]
        #
        # Extension:
        #   it is possible to use tlags=[-3,-2,-1,0,1]
        #   in this case, it is necessary to start with the position '3'
        #   and advance 'i' ONLY of 2 slots.
        #   Really usable slots: [0,1]
        assert isinstance(i, (int, np.int32)), "The argument 'i' must be the location update (an integer)"

        tlags = self.tlags if t is None else [t]
        tstart = self.tstart if t is None else 0

        st = len(tlags)         # length of tlags
        mt = max(tlags)         # max tlags index
        nfh = len(self.yp)      # length of fh
        my = self.yp.shape[1]   # predicted data size |y[i]|

        # convert y_pred as a 3D tensor
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((1, 1, my))
        elif len(y_pred.shape) == 2:
            y_pred = y_pred.reshape((1, -1, my))
        assert len(y_pred.shape) == 3

        for j in range(tstart, st):
            k = i + tlags[j]
            if k < nfh:
                try:
                    self.yp[k] = y_pred[0, j]
                except IndexError:
                    pass

        return i + mt + 1
    # end

    def fit_transform(self, y, X=None, fh=None):
        # It is not correct to use 'fit_transform(y, X)' because
        # 1) with 'fit(y,X)' it is passed y_train and X_train
        # 2) with 'transform(fh, X)' it is passed the forecasting horizon and X_prediction
        # that is, different information
        raise NotImplemented()

    # def to_pandas(self, yp):
    #     if not isinstance(self.y_pandas, (pd.DataFrame, pd.Series)):
    #         return yp
    #
    #     n = len(yp)
    #     cutoff = self.y_pandas.index[-1]
    #     fh = ForecastingHorizon(lrange(1, n+1), is_relative=True).to_absolute(cutoff)
    #     if isinstance(self.y_pandas, pd.DataFrame):
    #         return pd.DataFrame(data=yp, columns=self.y_pandas.columns, index=fh.to_pandas())
    #     else:
    #         return pd.Series(data=yp, name=self.y_pandas.name, index=fh.to_pandas())
# end


# ---------------------------------------------------------------------------
#   End
# ---------------------------------------------------------------------------
