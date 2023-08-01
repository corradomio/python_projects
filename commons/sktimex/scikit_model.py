import logging

from sktime.forecasting.base import BaseForecaster

from .forecasting.compose import make_reduction
from .lag import LagSlots, resolve_lag
from .utils import *


# ---------------------------------------------------------------------------
# ScikitForecastRegressor
# ---------------------------------------------------------------------------

def _replace_lags(kwargs: dict) -> dict:
    if "lags" in kwargs:
        lags = kwargs["lags"]
        del kwargs["lags"]
    else:
        lags = None

    if "current" in kwargs:
        current = kwargs["current"]
        del kwargs["current"]
    else:
        current = False

    if lags is not None:
        rlags: LagSlots = resolve_lag(lags, current)
        window_length = len(rlags)
        kwargs["window_length"] = window_length
    # end
    return kwargs
# end


class ScikitForecastRegressor(BaseForecaster):

    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.DataFrame",
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

    def __init__(self,
                 class_name: str = "sklearn.linear_model.LinearRegression",
                 y_only: bool = False,
                 **kwargs):
        super().__init__()

        kwargs = _replace_lags(kwargs)

        self._class_name = class_name
        self._kwargs = kwargs
        self._y_only = y_only
        self._fh_fit = None

        model_class = import_from(class_name)

        # extract the top namespace
        p = class_name.find('.')
        ns = class_name[:p]
        if ns in SCIKIT_NAMESPACES:
            window_length = kwval(kwargs, "window_length", 5)
            strategy = kwval(kwargs, "strategy", "recursive")

            kwargs = dict_del(kwargs, ["window_length", "strategy"])
            # create the regressor
            regressor = model_class(**kwargs)
            # create the forecaster
            self.forecaster = make_reduction(regressor, window_length=window_length, strategy=strategy)
        elif ns in SKTIME_NAMESPACES:
            # create the forecaster
            self.forecaster = model_class(**kwargs)
        else:
            # raise ValueError(f"Unsupported class_name '{class_name}'")
            pass

        p = class_name.rfind('.')
        self._log = logging.getLogger(f"ScikitForecastRegressor.{class_name[p+1:]}")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------
    
    @property
    def strategy(self):
        return kwval(self._kwargs, "strategy", "recursive")

    def get_params(self, deep=True):
        params = {} | self._kwargs
        params['class_name'] = self._class_name
        params['y_only'] = self._y_only
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
    #                       == fit(y, X[:y])
    #                          predict(fh, X[y:]
    #

    def _fit(self, y, X=None, fh: FH_TYPES = None):
        # ensure fh relative AND not None for tabular models
        fh = self._make_fh_relative(fh)
        self._fh_fit = fh

        if self._y_only: X = None
        
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self
    # end

    def _make_fh_relative(self, fh):
        # if is_tabular and gh is not defined, compose it based on 'window_length'
        is_tabular = 'window_length' in self._kwargs and 'strategy' in self._kwargs
        if fh is None and is_tabular:
            window_length = self._kwargs['window_length']
            fh = ForecastingHorizon(list(range(1, window_length+1)))
        if fh is None or fh.is_relative:
            return fh
        else:
            return fh.to_relative(self.cutoff)
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> pd.DataFrame:
        # WARN: fh must be a ForecastingHorizon
        assert isinstance(fh, ForecastingHorizon)

        if self._y_only: X = None

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        fh = fh.to_relative(self.cutoff)

        # using 'sktimex.forecasting.compose.make_reduction'
        # it is resolved the problems with predict horizon larger than the train horizon

        y_pred = self.forecaster.predict(fh=fh, X=X)

        assert isinstance(y_pred, (pd.DataFrame, pd.Series))
        return y_pred
    # end

    def _update(self, y, X=None, update_params=True):
        if self._y_only: X = None

        return super()._update(y=y, X=X, update_params=update_params)
    # end

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
    # score
    # -----------------------------------------------------------------------

    def predict(self, fh=None, X=None, y=None):
        return super().predict(fh=fh, X=X)

    def score(self, y, X=None, fh=None) -> float:
        return super().score(y=y, X=X, fh=fh)

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state
    # end

    def __repr__(self):
        return f"SklearnForecastRegressor[{self.forecaster}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

