
__all__ = [
    'BaseForecaster',
    'KwArgsForecaster',
    'TransformForecaster'
]

from typing import Union

import numpy as np
import pandas as pd

import numpyx.scalers as nxscal
from sktime.forecasting.base import BaseForecaster as Sktime_BaseForecaster
from ...utils import NoneType, to_matrix
from ...utils import is_instance, kwexclude, kwval, as_dict


#
# Base classes for all 'sktimex' forecasters
#
#   sktime.forecasting.base.BaseForecaster
#       sktime.forecasting.ExtendedBaseForecaster
#           sktime.forecasting.TransformForecaster
#

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_index_frequency(index):
    assert isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)), \
        "Dataframe index not a DatetimeIndex, PeriodIndex, RangeIndex"

    if isinstance(index, pd.PeriodIndex) and index.freq is None:
        raise ValueError("PeriodIndex index without freq")
    elif isinstance(index, pd.DatetimeIndex) and index.freq is None:
        raise ValueError("DatetimeIndex index without freq")


# ---------------------------------------------------------------------------
# ExtendedBaseForecaster
# ---------------------------------------------------------------------------

class BaseForecaster(Sktime_BaseForecaster):
    """
    Base class for the new forecasters.
    It add the method

        predict_history(fh, X, yh, Xh)

    to predict selecting a generic period

    It reimplements the method 'predict(...)' to
    resolve a problem with the correct index value assigned
    to the predictions
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self):
        super().__init__()

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Copy & paste of the parent implementation to change just
    # how y_pred is converted into y_out
    # -----------------------------------------------------------------------

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=fh)

    # def _fit(self, y, X=None):
    #     raise NotImplementedError("abstract method")

    def _check_X_y(self, X=None, y=None):
        if y is None:
            return super()._check_X_y(X=X, y=y)

        # if a Pandas object is converted into a numpy array,
        # to convert back the array in a Pandas object it is necessary
        # to know the type of the index and the last value.
        # Select an element of an Pandas index, return a 'mini index'
        # containing the selected value. For this reason it is not
        # necessary to save the info about the index type but it is
        # enough to save the last value.
        # Steps:
        #
        #   1) check if the index has a 'freq' (mandatory)
        #   2) retrieve the ORIGINAL cutoff
        #   3) call the parent method
        #   4) save the cutoff in '_y_metadata'. (a)
        #
        # (a) '_y_metadata' it is created by 'super()._check_X_y(X=X, y=y)'

        check_index_frequency(y.index)

        y_cutoff = y.index[-1]
        Xy_inner = super()._check_X_y(X=X, y=y)
        self._y_metadata["y_cutoff"] = y_cutoff
        return Xy_inner

    def predict(self, fh=None, X=None, yh=None, Xh=None, update_params=False):
        if yh is not None:
            self.update(y=yh, X=Xh, update_params=update_params)

        # handle inputs
        self.check_is_fitted()

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict if no looping/vectorization needed
        if not self._is_vectorized:
            y_pred = self._predict(fh=fh, X=X_inner)
        else:
            # otherwise we call the vectorized version of predict
            y_pred = self._vectorize("predict", X=X_inner, fh=fh)

        #
        # HERE is where the implementation is changed!
        #
        # convert to output mtype, identical with last y mtype seen
        # y_out = convert_to(
        #     y_pred,
        #     self._y_metadata["mtype"],
        #     store=self._converter_store_y,
        #     store_behaviour="freeze",
        # )
        #
        y_out = self._convert_to(y_pred)

        return y_out

    def _convert_to(self, y_pred):
        #
        # HERE is where the implementation is changed!
        #
        # Used a 'bugfix version' fo 'convert_to' to resolve
        # the problem about the Seris/DataFrame index
        #
        from ...datatypes.convert import convert_to
        y_out = convert_to(
            y_pred,
            self._y_metadata["mtype"],
            store=self._converter_store_y,
            store_behaviour="freeze",
            y_cutoff=self._y_metadata["y_cutoff"],
        )

        return y_out

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    # def _update_fit(self, y, X):
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

    # def _fit(self, y, X=None, fh=None):
    #     raise NotImplementedError("abstract method")

    # def _predict(self, fh, X=None):
    #     raise NotImplementedError("abstract method")

    # def def _update(self, y, X=None, update_params=True):
    #     return super()._update(y, X=X, update_params=update_params)

    def predict_history(self, fh, X=None, yh=None, Xh=None, update_params=False):
        if yh is not None:
            self.update(y=yh, X=Xh, update_params=update_params)
        return self.predict(fh=fh, X=X)

    # -----------------------------------------------------------------------
    # IO
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# KwArgsForecaster
# ---------------------------------------------------------------------------

class KwArgsForecaster(BaseForecaster):
    """
    Save in 'self.kwargs' all keyword arguments
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **params):
        self.kwargs = params
        return self


# ---------------------------------------------------------------------------
# TransformForecaster
# ---------------------------------------------------------------------------
#
#   scaler = {
#       method=...
#       **extra_params
#   }

class TransformForecaster(BaseForecaster):
    """
    Apply automatically a (de)normalization to all data based on 'method'.
    For now, it is supported only the method "minmax", 'standard', 'identity',
    or None ('identity')

    Setting:

        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",

    The values passed to forecasters derived from this one will be 1D/2D numpy arrays
    """
    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "both",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

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
            return nxscal.MinMaxScaler(**method_args)
        elif method == 'standard':
            return nxscal.StandardScaler(**method_args)
        else:
            return nxscal.IdentityScaler(**method_args)
    # end

    def transform(self, y, X):
        # the first time it is used to create and train the scalers.
        # the second time the scalers are used as is
        #
        # DONT' remove: it is used to have y and X of the
        # same type (float32)
        y = to_matrix(y)
        X = to_matrix(X)

        assert isinstance(y, (NoneType, np.ndarray))
        assert isinstance(X, (NoneType, np.ndarray))

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

    def inverse_transform(self, y, X_pred=None):
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

