from datetime import datetime
from typing import Union, Any, Optional

import pandas as pd
import numpy as np
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from stdlib import qualified_name, import_from, is_instance
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from ...forecasting.base import BaseForecaster, KwArgsForecaster
from darts.timeseries import TimeSeries


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def to_timeseries(x: Union[None, pd.Series, pd.DataFrame, np.ndarray],
                  freq: Optional[str] = None) \
        -> Optional[TimeSeries]:

    assert is_instance(x, Union[None, pd.Series, pd.DataFrame, np.ndarray])
    assert is_instance(freq, Optional[str])

    def reindex(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        index = x.index
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
            if isinstance(x, pd.Series):
                x = pd.Series(data=x.values, index=index, name=x.name, dtype=x.dtype)
            else:
                x = x.set_index(index)
        return x

    ts: Optional[TimeSeries] = None
    if x is None:
        pass
    elif isinstance(x, pd.Series):
        ts = TimeSeries.from_series(reindex(x), freq=x.index.freqstr)
    elif isinstance(x, pd.DataFrame):
        ts = TimeSeries.from_dataframe(reindex(x), freq=x.index.freqstr)
    elif isinstance(x, np.ndarray):
        ts = TimeSeries.from_values(x)
    else:
        raise ValueError(f"Unsupported data type {type(x)}")

    return ts


def from_timeseries(X: TimeSeries, to_type) -> Any:
    if to_type == pd.Series:
        return X.pd_series()
    elif to_type == pd.DataFrame:
        return X.pd_dataframe()
    elif to_type == np.ndarray:
        return X.values()
    else:
        raise ValueError(f"Unsupported type {to_type}")


def fh_relative(fh: ForecastingHorizon, cutoff: datetime):
    if not fh.is_relative:
        fh = fh.to_relative(cutoff)
    return fh


# ---------------------------------------------------------------------------
# DartsBaseForecaster
# ---------------------------------------------------------------------------

class BaseDartsForecaster(KwArgsForecaster):

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

    def __init__(self, darts_model, kwargs):
        super().__init__(**kwargs)

        self.darts_model = qualified_name(darts_model)

        # for k in kwargs:
        #     setattr(self, k, kwargs[k])

        # Note: the models can be NOT created here because
        # some of them depend on the data!
        self._darts_class = import_from(self.darts_model)
        return

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):

        self._y_type = type(y)
        yts = to_timeseries(y)

        past_covariates = to_timeseries(X)

        kwargs = self._compose_kwargs(y, X, fh)
        self._model: GlobalForecastingModel = self._darts_class(**kwargs)

        if past_covariates is None:
            self._model.fit(yts)
        else:
            self._model.fit(yts, past_covariates=past_covariates)

        return self

    def _compose_kwargs(self, y, X, fh):
        return self.kwargs

    def _predict(self, fh: ForecastingHorizon, X=None):

        if self._X is not None and X is not None:
            X = pd.concat([self._X, X], axis='rows')
        elif X is not None:
            pass
        elif self._X is not None:
            X = self._X

        past_covariates = to_timeseries(X)
        nfh = len(fh)

        if past_covariates is None:
            ts_pred: TimeSeries = self._model.predict(nfh)
        else:
            ts_pred: TimeSeries = self._model.predict(
                nfh,
                past_covariates=past_covariates,
            )

        y_pred = from_timeseries(ts_pred, self._y_type)
        return y_pred

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
# end
