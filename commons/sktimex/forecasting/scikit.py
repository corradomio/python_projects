
__all__ = [
    'ScikitForecaster',
]

import logging
import pandas as pd
from typing import Union

from sktime.forecasting.base import ForecastingHorizon
from .base import KwArgsForecaster
from ..forecasting.compose import make_reduction
from ..utils import SKTIME_NAMESPACES, SCIKIT_NAMESPACES, FH_TYPES, PD_TYPES
from ..utils import import_from, kwval, kwexclude, qualified_name, ns_of, name_of


# ---------------------------------------------------------------------------
# ScikitForecaster
# ---------------------------------------------------------------------------

class ScikitForecaster(KwArgsForecaster):
    """
    sktime's forecaster equivalent to 'make_reduction(...)'

    For a more extended version of this class, see 'LinearForecaster'
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
    # can be passed:
    #
    #   1) a sktime  class name -> instantiate it
    #   2) a sklearn class name -> instantiate it and wrap it with make_reduction
    #   3) a sktime  instance   -> as is
    #   4) a sklearn instance   -> wrap it with make_reduction

    def __init__(self, *,
                 estimator: Union[str, type] = "sklearn.linear_model.LinearRegression",
                 window_length=5,
                 prediction_length=1,
                 **kwargs):

        super().__init__(**kwargs)

        assert isinstance(estimator, Union[str, type])

        # Unmodified parameters [readonly]
        self.estimator = qualified_name(estimator)
        self.window_length = window_length
        self.prediction_length = prediction_length

        # Effective parameters
        self._estimator = None
        # self._kwargs = _replace_lags(kwargs)

        self._create_estimator(kwargs)

        name = name_of(self.estimator)
        self._log = logging.getLogger(f"ScikitForecaster.{name}")
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

    def _fit(self, y, X=None, fh: FH_TYPES = None):
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

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> pd.DataFrame:
        # WARN: fh must be a ForecastingHorizon
        assert isinstance(fh, ForecastingHorizon)

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        # fh = fh.to_relative(self.cutoff)
        nfh = len(fh)
        efh = ForecastingHorizon(list(range(1, nfh+1)))

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
ScikitForecastRegressor = ScikitForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
