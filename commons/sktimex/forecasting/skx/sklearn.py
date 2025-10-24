
__all__ = [
    'ScikitLearnForecaster',
]

import logging
import pandas as pd
from typing import Union

from sktime.forecasting.base import ForecastingHorizon
from ..base import BaseForecaster
from ...forecasting.compose import make_reduction
from ...utils import SKTIME_NAMESPACES, SCIKIT_NAMESPACES, FH_TYPES, PD_TYPES
from stdlib import import_from, kwval, kwexclude, qualified_name, ns_of, name_of


# ---------------------------------------------------------------------------
# ScikitForecaster
# ---------------------------------------------------------------------------

class ScikitLearnForecaster(BaseForecaster):

    """
    sktime's forecaster equivalent to 'make_reduction(...)'

    For a more extended version of this class, see 'LinearForecaster'
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    # can be passed:
    #
    #   1) a sktime  class name -> instantiate it
    #   2) a sklearn class name -> instantiate it and wrap it with make_reduction
    #   3) a sktime  instance   -> as is
    #   4) a sklearn instance   -> wrap it with make_reduction

    def __init__(self, *,
                 estimator: Union[str, type] = "sklearn.linear_model.LinearRegression",
                 estimator_args=None,
                 window_length=5,
                 prediction_length=1):

        super().__init__(**estimator_args)

        assert isinstance(estimator, Union[str, type])

        # Unmodified parameters [readonly]
        self.estimator = qualified_name(estimator)
        self.estimator_args = estimator_args
        self.window_length = window_length
        self.prediction_length = prediction_length

        # Effective parameters
        self._estimator = None
        # self._kwargs = _replace_lags(kwargs)

        self._create_estimator(estimator_args or {})

        name = name_of(self.estimator)
        self._log = logging.getLogger(f"sktimex.ScikitLearnForecaster.{name}")
    # end

    def _create_estimator(self, kwargs):
        estimator = import_from(self.estimator)

        ns = ns_of(self.estimator)
        if ns in SCIKIT_NAMESPACES:
            window_length = self.window_length
            # window_length = kwval(kwargs, "window_length", 5)
            strategy = kwval(kwargs, "strategy", "recursive")
            kwargs = kwexclude(kwargs, ["window_length", "strategy"])

            # create the scikit-learn regressor
            regressor = estimator(**kwargs)
            # create the forecaster
            self._estimator = make_reduction(regressor, window_length=window_length, strategy=strategy)
        elif ns in SKTIME_NAMESPACES:
            # create a sktime forecaster
            self._estimator = estimator(**kwargs)
        else:
            raise ValueError(f"Unsupported estimator '{estimator}'")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | {
            'estimator': self.estimator,
            'prediction_length': self.prediction_length,
            'window_length': self.window_length
        }
        return params

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
    #                       == fit(y, X[:y])
    #                          predict(fh, X[y:])
    #

    def _fit(self, y, X, fh):
        # if fh is None, it uses self.prediction_length
        # Note: prediction length

        # ensure fh relative AND NOT None for tabular models
        fh = self._compose_tabular_fh(fh)

        self._estimator.fit(y=y, X=X, fh=fh)

        return self
    # end

    def _compose_tabular_fh(self, fh):
        # ensure fh relative AND NOT None for tabular models
        if fh is None and self.prediction_length is None:
            fh = ForecastingHorizon([1], is_relative=True)
        elif isinstance(fh, int):
            pl = fh
            fh = ForecastingHorizon(list(range(1, pl + 1)), is_relative=True)
        elif isinstance(fh, ForecastingHorizon):
            fh = fh if fh.is_relative else fh.to_relative(self.cutoff)
        elif self.prediction_length >= 1:
            pl = self.prediction_length
            fh = ForecastingHorizon(list(range(1, pl + 1)), is_relative=True)
        else:
            raise ValueError(f"Unsupported fh {fh}")
        return fh
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> Union[pd.DataFrame, pd.Series]:
        super()._predict(fh, X)

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        # fh = fh.to_relative(self.cutoff)
        # nfh = len(fh)
        # efh = ForecastingHorizon(list(range(1, nfh+1)))
        efh = fh.to_relative(self.cutoff)

        # using 'sktimex.forecasting.compose.make_reduction'
        # it is resolved the problems with predict horizon larger than the train horizon

        y_pred: pd.Series = self._estimator.predict(fh=efh, X=X)

        # assert isinstance(y_pred, (pd.DataFrame, pd.Series))
        index = fh.to_absolute(self.cutoff).to_pandas()
        y_pred.index = index
        # y_pred = pd.Series(data=y_pred.values, index=index)
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

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    def __repr__(self):
        return f"ScikitForecaster[{self.estimator}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# Compatibility
ScikitLearnForecastRegressor = ScikitLearnForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
