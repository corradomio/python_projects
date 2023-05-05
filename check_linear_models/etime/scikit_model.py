import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction

from stdlib import import_from, dict_del, kwval


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']


# ---------------------------------------------------------------------------
# ScikitForecastRegressor
# ---------------------------------------------------------------------------

class ScikitForecastRegressor:

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 **kwargs):

        model_class = import_from(class_name)

        # extract the top namespace
        p = class_name.find('.')
        ns = class_name[:p]
        if ns in SCIKIT_NAMESPACES:
            window_length = kwval(kwargs, 'window_length', 5)
            strategy = kwval(kwargs, 'strategy', 'recursive')

            kwargs = dict_del(kwargs, ['window_length', 'strategy'])

            regressor = model_class(**kwargs)
            self.forecaster = make_reduction(regressor, window_length=window_length, strategy=strategy)
        elif ns in SKTIME_NAMESPACES:
            self.forecaster = model_class(**kwargs)
        else:
            raise ValueError(f"Unsupported class_name '{class_name}'")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def cutoff(self):
        return self.forecaster.cutoff

    @property
    def fh(self):
        return self.forecaster.fh

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def fit(self, y, X=None, fh=None):
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self

    def predict(self, fh: ForecastingHorizon = None, X=None, y=None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select
        # the WRONG rows.
        fh = fh.to_relative(self.forecaster.cutoff).to_numpy()
        if X is not None:
            n = len(X)
        else:
            n = fh[-1]
        fhr = ForecastingHorizon(np.arange(1, n + 1))
        y_pred = self.forecaster.predict(fh=fhr, X=X)
        return y_pred.iloc[fh-1]

    # def fit_predict(self, y, X=None, fh=None):
    #     return self.forecaster.fit_predict(y=y, X=X, fh=fh)
    #
    # def score(self, y, X=None, fh=None):
    #     return self.forecaster.score(y=y, X=X, fh=fh)

    # -----------------------------------------------------------------------

    # def predict_quantiles(self, fh=None, X=None, alpha=None):
    #     return self.forecaster.predict_quantiles(fh=fh, X=X, alpha=alpha)
    #
    # def predict_interval(self, fh=None, X=None, coverage=0.90):
    #     return self.forecaster.predict_interval(fh=fh, X=X, coverage=coverage)
    #
    # def predict_var(self, fh=None, X=None, cov=False):
    #     return self.forecaster.predict_var(fh=fh, X=X, cov=cov)
    #
    # def predict_proba(self, fh=None, X=None, marginal=True):
    #     return self.forecaster.predict_proba(fh=fh, X=X, marginal=marginal)
    #
    # def predict_residuals(self, y=None, X=None):
    #     return self.forecaster.predict_residuals(y=y, X=X)

    # -----------------------------------------------------------------------

    # def update(self, y, X=None, update_params=True):
    #     self.forecaster.update(y=y, X=X, update_params=update_params)
    #     return self
    #
    # def update_predict(self, y, cv=None, X=None, update_params=True, reset_forecaster=True):
    #     self.forecaster.update_predict(y=y, cv=cv, X=X, update_params=update_params,
    #                                    reset_forecaster=reset_forecaster)
    #     return self
    #
    # def update_predict_single(self, y=None, fh=None, X=None, update_params=True):
    #     self.forecaster.update_predict_single(y=y, fh=fh, X=X, update_params=update_params)
    #     return self

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"SklearnForecastRegressor[{self.forecaster}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end

