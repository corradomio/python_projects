__all__ = [
    'BaseForecaster',
    'ScaledForecaster'
]

from typing import Union

import pandas as pd
from sktime.forecasting.base import BaseForecaster as Sktime_BaseForecaster, ForecastingHorizon

import pandasx as pdx
from ...utils import NoneType
from ...utils import kwexclude, kwval, as_dict


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_index_frequency(index):
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
        raise ValueError("Dataframe index not a DatetimeIndex, PeriodIndex, RangeIndex")
    elif isinstance(index, pd.PeriodIndex) and index.freq is None:
        raise ValueError("PeriodIndex index without freq")
    elif isinstance(index, pd.DatetimeIndex) and index.freq is None:
        raise ValueError("DatetimeIndex index without freq")


# ---------------------------------------------------------------------------
# BaseForecaster
# ---------------------------------------------------------------------------

class BaseForecaster(Sktime_BaseForecaster):

    _tags = {
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "scitype:y": "both",
        # "ignores-exogeneous-X": False,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": True
    }

    """
    Base class for the new forecasters.
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    # def __init__(self):
    #     super().__init__()

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Copy & paste of the parent implementation to change just
    # how y_pred is converted into y_out
    # -----------------------------------------------------------------------
    # Note: there is a problem converting a Pandas object to a numpy array and
    # back to a Pandas object.

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    # def fit(self, y, X=None, fh=None):
    #     ...

    # def predict(self, fh=None, X=None):
    #     ...

    # def fit_predict(self, y, X=None, fh=None):
    #     ...

    # def score(self, y, X=None, fh=None):
    #     ...

    # -----------------------------------------------------------------------

    # def predict_quantiles(self, fh=None, X=None, alpha=None):
    #     ...

    # def predict_interval(self, fh=None, X=None, coverage=0.90):
    #     ...

    # def predict_var(self, fh=None, X=None, cov=False):
    #     ...

    # def predict_proba(self, fh=None, X=None, marginal=True):
    #     ...

    # def predict_residuals(self, y=None, X=None):
    #     ...

    # -----------------------------------------------------------------------

    # def update(self, y, X=None, update_params=True):
    #     ...

    # def update_predict(self, y, cv=None, X=None, update_params=True, reset_forecaster=True):
    #     ...

    # def update_predict_single(self, y=None, fh=None, X=None, update_params=True):
    #     ...

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    # def _fit(self, y, X, fh):
    #     raise NotImplementedError("abstract method")

    def _predict(self, fh, X):
        assert isinstance(fh, ForecastingHorizon), "'fh' must be a ForecastingHorizon"

    # def _update(self, y, X=None, update_params=True):
    #     return super()._update(y, X=X, update_params=update_params)

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def _check_fh(self, fh, pred_int=False):
        """
        This fix an error in the fh conversion:
        IF fh is a pandas Index AND new_fh is "relative", change the
        flag into "absolute"
        """
        new_fh: ForecastingHorizon = super()._check_fh(fh, pred_int=pred_int)
        if isinstance(fh, pd.Index) and new_fh.is_relative:
            new_fh._is_relative = False
        return new_fh

# end

# ---------------------------------------------------------------------------
# TransformForecaster
# ---------------------------------------------------------------------------
#
#   scaler = {
#       method=...
#       **extra_params
#   }

class ScaledForecaster(BaseForecaster):
    """
    Apply automatically a (de)normalization to all data based on 'method'.
    For now, it is supported only the method "minmax", 'standard', 'identity',
    or None ('identity')

    Setting:

        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "scitype:y": "both",

    The values passed to forecasters derived from this one will be 1D/2D numpy arrays
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    # scaler = "minmax" | "standard"
    # scaler = {
    #     "method": "minmax",
    #     ...
    # }

    def __init__(self, *,
                 scaler=None,
                 # **kwargs
                 ):
        # super().__init__(**kwargs)
        super().__init__()

        assert isinstance(scaler, Union[NoneType, str, dict])

        # Unmodified parameters [readonly]
        self.scaler = scaler

        # Effective parameters
        self._X_scaler = None
        self._y_scaler = None
        self._scaler_params = as_dict(scaler, key="method")
        return

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    # def get_params(self, deep=True):
    #     params = super().get_params(deep=deep) | dict(
    #         scaler=self.scaler
    #     )
    #     return params

    # def set_params(self, **params):
    #     self.method = params['method']
    #     self.clip = params['clip']
    #     super().set_params(**kwexclude(params, ['method', 'clip']))
    #     return self

    # -----------------------------------------------------------------------
    # support
    # -----------------------------------------------------------------------

    def _create_scaler(self):
        method = kwval(self._scaler_params, key="method", defval=None)
        method_args = kwexclude(self._scaler_params, "method")
        if method == 'minmax':
            return pdx.MinMaxScaler(**method_args)
        elif method == 'standard':
            return pdx.StandardScaler(**method_args)
        else:
            return pdx.IdentityScaler(**method_args)
    # end

    def transform(self, y, X):
        # the first time it is used to create and train the scalers.
        # the second time the scalers are used as is
        #
        # DONT' remove: it is used to have y and X of the
        # same type (float32)
        # y = to_numpy(y, matrix=True)
        # X = to_numpy(X)

        assert isinstance(y, (NoneType, pd.DataFrame, pd.Series))
        assert isinstance(X, (NoneType, pd.DataFrame, pd.Series))

        if y is not None:
            if self._y_scaler is None:
                self._y_scaler = self._create_scaler()
                self._y_scaler.fit(y)
            y = self._y_scaler.transform(y)
        # end
        if X is not None:
            if self._X_scaler is None:
                self._X_scaler = self._create_scaler()
                self._X_scaler.fit(X)
            X = self._X_scaler.transform(X)
        return y, X
    # end

    def inverse_transform(self, y):
        if y is not None:
            y = self._y_scaler.inverse_transform(y)
        return y
    # end

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
