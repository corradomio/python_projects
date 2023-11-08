from typing import Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ..lags import LagSlots, resolve_lags, resolve_tlags
from ..utils import NoneType, to_matrix, lrange


# ---------------------------------------------------------------------------
#   ModelTransform
#       ModelTrainTransform
#       ModePredictTransform
# ---------------------------------------------------------------------------

class TimeseriesTransform:

    def _check_X(self, X):
        X = to_matrix(X)
        assert isinstance(X, (NoneType, np.ndarray))
        return X

    def _check_Xy(self, X, y=None):
        X = to_matrix(X)
        y = to_matrix(y)
        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(y, (NoneType, np.ndarray))
        return X, y

    def _check_Xfh(self, X, fh):
        # Note: to be compatible with 'sktime' fh MUST starts 1 timeslot after the
        # 'cutoff', that is, the LAST timestamp used in training. This means that
        # If specified as list of as a ForecastingHorizon, it MUST be:
        #
        #   [1,2,3.....]
        #
        # Then, the LAST value is the 'prediction_length'.
        #
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

        assert isinstance(X, (NoneType, np.ndarray))
        assert isinstance(fh, int)
        return X, fh
# end


class ModelTrainTransform(TimeseriesTransform):

    def __init__(self, slots, tlags=(0,)):
        if not isinstance(slots, LagSlots):
            slots = resolve_lags(slots)
        if isinstance(tlags, int):
            tlags = resolve_tlags(tlags)

        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list))

        self.slots = slots
        self.xlags: list = slots.xlags
        self.ylags: list = slots.ylags
        self.tlags: list = list(tlags)
    # end

    def fit(self, y: np.ndarray, X: Optional[np.ndarray]=None):
        X, y = self._check_Xy(X, y)
        return self
    # end

    def transform(self, y: np.ndarray, X: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._check_Xy(X, y)
        return X, y
    # end

    def fit_transform(self, y: np.ndarray, X: Optional[np.ndarray]=None):
        return self.fit(y=y, X=X).transform(y=y, X=X)
# end


class ModelPredictTransform(TimeseriesTransform):

    def __init__(self, slots, tlags=(0,)):
        if not isinstance(slots, LagSlots):
            slots = resolve_lags(slots)
        if isinstance(tlags, int):
            tlags = resolve_tlags(tlags)

        assert isinstance(slots, LagSlots)
        assert isinstance(tlags, (tuple, list)), f"Parameter tlags not of type list|tuple: {tlags}"

        self.slots: LagSlots = slots
        self.xlags: list[int] = slots.xlags
        self.ylags: list[int] = slots.ylags
        self.tlags: list[int] = tlags

        self.Xh = None  # X history
        self.yh = None  # y history

        self.Xt = None  # X transform
        self.yt = None  # y transform
        self.fh = None  # fh transform

        self.Xp = None  # X prediction
        self.yp = None  # y prediction
    # end

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        # used to support the implementation of 'self.to_pandas()'
        self.y_pandas = y

        # This method must be used to pass 'y_train' and 'X_train'
        X, y = self._check_Xy(X, y)

        self.Xh = X
        self.yh = y

        return self
    # end

    def transform(self, fh: int, X: Optional[np.ndarray] = None):
        # This method must e used with 'fh' and 'X_test'/'X_predict'
        X, fh = self._check_Xfh(X, fh)

        self.Xt = X
        self.fh = fh

        return fh, X
    # end

    def update(self, i, y_pred, t=None):
        tlags = [t] if t is not None else self.tlags
        st = len(tlags)
        mt = max(tlags)
        nfh = len(self.yp)
        my = self.yp.shape[1]

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((1, 1, my))
        elif len(y_pred.shape) == 2:
            y_pred = y_pred.reshape((1, -1, my))

        assert len(y_pred.shape) == 3

        for j in range(st):
            k = i + tlags[j]
            if k < nfh:
                self.yp[k] = y_pred[0, j]

        return i + mt + 1
    # end

    def fit_transform(self, y, X=None):
        # It is not correct to use 'fit_transform(y, X)' because
        # 1) with 'fit(y,X)' it is passed y_train and X_train
        # 2) with 'transform(fh, X)' it is passed the forecasting horizon and X_prediction
        # that is, different information
        raise NotImplemented()

    def to_pandas(self, yp):
        if not isinstance(self.y_pandas, (pd.DataFrame, pd.Series)):
            return yp

        n = len(yp)
        cutoff = self.y_pandas.index[-1]
        fh = ForecastingHorizon(lrange(1, n+1), is_relative=True).to_absolute(cutoff)
        if isinstance(self.y_pandas, pd.DataFrame):
            return pd.DataFrame(data=yp, columns=self.y_pandas.columns, index=fh.to_pandas())
        else:
            return pd.Series(data=yp, name=self.y_pandas.name, index=fh.to_pandas())
# end
