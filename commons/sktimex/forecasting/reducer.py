
__all__ = [
    'ReducerForecaster',
]

import logging
from typing import Sized, cast, Union, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .base import BaseForecaster
from ..forecasting.compose import make_reduction
from ..utils import PD_TYPES
from ..utils import import_from, qualified_name, lrange, name_of


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class ReducerForecaster(BaseForecaster):
    """
    sktime's forecaster equivalent to 'make_reduction(...)'

    For a more extended version of this class, see 'LinearForecaster'
    """
    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    # can be passed:
    #
    #   1) a sklearn class name -> instantiate it and wrap it with make_reduction
    #
    #   SUSPENDED: not very useful
    #
    #   2) a sktime  class name -> instantiate it                       NO
    #   3) a sktime  instance   -> as is                                NO
    #   4) a sklearn instance   -> wrap it with make_reduction          NO
    #
    # How to pass the estimator parameters?
    #
    #   1) inline:  <name>=<value>
    #   2) estimator_params={<name>: <value>, ...}
    #   3) estimator__<name> = <value>
    #
    # Note: 1), 3) require '**kwargs' an the a 'custom' 'get_params()/set_parame()'
    #

    def __init__(self, *,
                 estimator: Union[str, type] = "sklearn.linear_model.LinearRegression",
                 estimator_params: Optional[dict] = None,
                 window_length=10,
                 prediction_length=1,
                 strategy="recursive",
                 windows_identical=False):

        super().__init__()

        assert isinstance(estimator, Union[str, type])

        # Unmodified parameters [readonly]
        self.estimator = qualified_name(estimator)
        self.estimator_params = estimator_params
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.strategy = strategy
        self.windows_identical = windows_identical

        # Effective parameters
        self._estimator = None

        kwargs = estimator_params or {}
        self._create_estimator(kwargs)

        name = name_of(self.estimator)
        self._log = logging.getLogger(f"SklearnForecaster.{name}")
    # end

    def _create_estimator(self, kwargs):
        estimator = import_from(self.estimator)

        # create the scikit-learn regressor
        regressor = estimator(**kwargs)

        # create the forecaster
        self._estimator = make_reduction(
            regressor,
            window_length=self.window_length,
            prediction_length=self.prediction_length,
            strategy=self.strategy,
            windows_identical=self.windows_identical
        )
        return self
    # end

    # -----------------------------------------------------------------------
    # fit/predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)       fit(y, X, fh)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
    #                       == fit(y, X[:y])
    #                          predict(fh, X[y:])
    #

    def _fit(self, y, X=None, fh=None):
        assert isinstance(y, np.ndarray)

        # ensure fh relative AND NOT None for tabular models
        # Note: these forecasters require fh in fit
        fh_in_fit = self._compose_fh_in_fit(fh)

        self._estimator.fit(y=y, X=X, fh=fh_in_fit)

        return self
    # end

    def _compose_fh_in_fit(self, fh):
        # 'fh' in fit is used to generate predictions for
        # the complete 'prediction_horizon'

        if fh is None and self.prediction_length is None:
            fh = ForecastingHorizon([1])
        elif isinstance(fh, int):
            pl = fh
            fh = ForecastingHorizon(lrange(1, pl+1))
        elif isinstance(fh, ForecastingHorizon):
            fh = fh if fh.is_relative else fh.to_relative(self.cutoff)
        elif self.prediction_length >= 1:
            pl = self.prediction_length
            fh = ForecastingHorizon(lrange(1, pl + 1))
        else:
            raise ValueError(f"Unsupported fh {fh}")
        assert fh.is_relative
        return fh
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> pd.DataFrame:
        # WARN: fh must be a ForecastingHorizon
        assert isinstance(fh, ForecastingHorizon)

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        # fh = fh.to_relative(self.cutoff)
        nfh = len(cast(Sized, fh))
        efh = ForecastingHorizon(lrange(1, nfh+1))

        # using 'sktimex.forecasting.compose.make_reduction'
        # it is resolved the problems with predict horizon larger than the train horizon

        y_pred = self._estimator.predict(fh=efh, X=X)

        # assert isinstance(y_pred, (pd.DataFrame, pd.Series))
        return y_pred
    # end

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def _update(self, y, X=None, update_params=True):
        try:
            self._estimator.update(y=y, X=X, update_params=False)
        except:
            pass
        return super()._update(y=y, X=X, update_params=False)

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"ReducerForecaster[{name_of(self.estimator)}, {self.strategy}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end
