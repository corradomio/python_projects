from typing import Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from sktimex.forecasting.base import ScaledForecaster
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    return None if df is None else df.to_numpy()


def _from_numpy(arr: np.ndarray, index, template: pd.Series) -> pd.Series:
    return pd.Series(data=arr,index=index,name=template.name)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _setattr(obj, a, v):
    if a not in ["scaler", "h"]:
        assert not hasattr(obj, a), f"Attribute {a} already present"
    setattr(obj, a, v)

def _parse_kwargs(kwargs: dict) -> dict:
    stf_kwargs = {}|kwargs

    return stf_kwargs

# ---------------------------------------------------------------------------

class _BaseStatsForecastForecaster(ScaledForecaster):

    # Each derived class must have a specifalized set of tags

    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
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
        # "ignores-exogeneous-X": False,
        "capability:exogenous": True,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # -----------------------------------------------------------------------

    def __init__(self, stf_class: type, locals: dict):
        super().__init__(scaler=locals.get("scaler", None))

        self._stf_class = stf_class

        self.verbose = False

        self._model = None
        self._kwargs = {}
        self._prediction_intervals = None

        self._ignores_exogenous_X = not self.get_tag("capability:exogenous", False)
        self._future_exogenous_X = self.get_tag("capability:future-exogenous", False, False)
        self._analyze_locals(locals)
        pass
    # end

    def _analyze_locals(self, locals: dict):
        assert "kwargs" not in locals

        keys = list(locals.keys())
        for k in keys:
            if k in ["self", "__class__"]:
                del locals[k]
            elif k == "verbose":
                self.verbose = locals[k]
                del locals[k]
            else:
                _setattr(self, k, locals[k])

        self._kwargs = locals
    # end

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    def get_param_names(self, sort=True):
        param_names = list(self._kwargs.keys()) + ["verbose"]
        if sort:
            param_names = sorted(param_names)
        return param_names

    # def get_params(self, deep=True):
    #     return super().get_params(deep=deep)
    def get_params(self, deep=True):
        params = {}
        params |= super().get_params(deep=deep)
        params |= self._kwargs | {"verbose": self.verbose}
        return params

    def set_params(self, **params):
        super_params = {}
        for k in params:
            if k == "scaler":
                super_params[k] = params[k]
            elif k == "verbose":
                self.verbose = params[k]
            else:
                self._kwargs[k] = params[k]
        return super().set_params(**super_params)

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Compatibility with neuraforecast:
    # darts: the parameters for pl.Trainer are saved in 'pl_trainer_kwargs'
    #    nf: the parameters for pl.Trainer are flat

    def _compile_model(self, y, X):

        # create the StatsForecast parameters (cloned)
        stf_kwargs = _parse_kwargs(self._kwargs)

        stf_kwargs["class"] = self._stf_class

        # apply special transformations to the parameters
        stf_kwargs = self._validate_kwargs(stf_kwargs, y, X)

        model = create_from(stf_kwargs)

        return model
    # end

    def _validate_kwargs(self, stf_kwargs: dict, y, X) -> dict:
        return stf_kwargs
    # end

    # -----------------------------------------------------------------------

    def _fit(self, y, X, fh):

        # y, X = self.transform(y, X)

        # create the model
        self._model = self._compile_model(y, X)

        y_arr = _to_numpy(y)
        X_arr = _to_numpy(None if self._ignores_exogenous_X else X)

        self._model.fit(y_arr, X_arr)

        return self

    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X):
        X_arr = _to_numpy(None if self._ignores_exogenous_X else X)
        nfh = len(fh)

        y_pred_dict = self._model.predict(h=nfh, X=X_arr)
        y_pred_arr = y_pred_dict["mean"]

        index = fh.to_absolute_index(self._cutoff)
        return _from_numpy(y_pred_arr, index, self._y)

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
# end
