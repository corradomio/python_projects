from datetime import datetime
from typing import Union

import pandas as pd
import numpy as np
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from ..forecasting.base import ExtendedBaseForecaster
from darts.timeseries import TimeSeries

# Index
#   CFTimeIndex
#   ExtensionIndex
#       IntervalIndex
#       NDArrayBackendExtensionIndex
#           CategoricalIndex
#           DatetimeIndexOpsMixin
#               DatetimeTimedeltaMixin
#                   DatetimeIndex
#                   TimedeltaIndex
#               PeriodIndex
#       MultiIndex
#       RangeIndex
# .


def to_timeseries(X: Union[pd.Series, pd.DataFrame, np.ndarray]) -> TimeSeries:
    # supported ONLY RangeIndex, DatetimeIndex
    if isinstance(X, (pd.DataFrame, pd.Series)):
        index: Union[pd.Index] = X.index
        if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex)):
            freq = index.freq
            index = X.index
        elif isinstance(index, pd.PeriodIndex):
            freq = index.freq
            index = index.to_timestamp()
        # elif isinstance(index, pd.IntervalIndex):
        #     pass
        # elif isinstance(index, pd.TimedeltaIndex):
        #     pass
        else:
            raise ValueError(f"Unsupported index {type(index)}")
        if isinstance(X, pd.Series):
            X = X.reindex(index=index)
        else:
            X = X.set_index(index)
        pass
    # end
    if isinstance(X, pd.DataFrame):
        return TimeSeries.from_dataframe(X)
    if isinstance(X, pd.Series):
        return TimeSeries.from_series(X, freq=freq)
    if isinstance(X, np.ndarray):
        return TimeSeries.from_values(X, freq=freq)
    else:
        raise ValueError(f"Unsupported type: {type(X)}")


def fh_relative(fh: ForecastingHorizon, cutoff: datetime):
    if not fh.is_relative:
        fh = fh.to_relative(cutoff)
    return fh


class DartsBaseForecaster(ExtendedBaseForecaster):

    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
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
        "scitype:y": "univariate",
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

    def __init__(self, mclass, pargs, kwargs):
        super().__init__()
        self._mclass = mclass
        self._pargs = pargs
        self._kwargs = kwargs
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self._ptype = None

    # -----------------------------------------------------------------------

    def _create_model(self):
        return self._mclass(*self._pargs, **self._kwargs)

    def _fit(self, y, X=None, fh=None):
        self._ptype = type(y)
        self._model = self._create_model()
        yts = self.to_ts(y)
        self._model.fit(yts)
        return self

    def _predict(self, fh: ForecastingHorizon, X=None):
        model = self._model
        nfh = len(fh)
        pts: TimeSeries = model.predict(nfh)
        return self.from_ts(pts)

    # -----------------------------------------------------------------------

    def to_ts(self, X):
        return to_timeseries(X)

    def from_ts(self, ts: TimeSeries):
        if self._ptype == pd.Series:
            return ts.pd_series()
        elif self._ptype == pd.DataFrame:
            return ts.pd_dataframe()
        elif self._ptype == np.ndarray:
            return ts.values()
        else:
            raise ValueError(f"Unsupported type {self._ptype}")
# end
