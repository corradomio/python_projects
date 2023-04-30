from typing import Optional

import numpy as np
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.compose import make_reduction

from stdlib import import_from, dict_del, kwval


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKLEARN_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']


# ---------------------------------------------------------------------------
# ScikitForecastRegressor
# ---------------------------------------------------------------------------

class ScikitForecastRegressor(BaseForecaster):

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "X_inner_mtype": ["pd.DataFrame"],
        "y_inner_mtype": ["pd.DataFrame"],
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 **kwargs):
        super().__init__()

        self._class_name = class_name
        self._kwargs = {} | kwargs

        model_class = import_from(class_name)

        p = class_name.find('.')
        top_ns = class_name[:p]
        if top_ns in SKLEARN_NAMESPACES:
            #
            # sklearn.* class
            #
            window_length = kwval(kwargs, 'window_length', 5)
            reduction_strategy = kwval(kwargs, 'strategy', 'recursive')

            kwargs = dict_del(kwargs, ['window_length', 'strategy'])

            regressor = model_class(**kwargs)
            self.forecaster = make_reduction(regressor, window_length=window_length, strategy=reduction_strategy)
        elif top_ns == 'sktime':
            #
            # sktime class
            #
            self.forecaster = model_class(**kwargs)
        else:
            raise ValueError(f"Unsupported class {class_name}")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    # @property
    # def cutoff(self):
    #     return self.forecaster.cutoff

    # @property
    # def fh(self):
    #     return self.forecaster.fh

    def get_params(self, deep=True):
        return dict(
            class_name=self._class_name,
            **self._kwargs
        )

    # def get_tags(self):
    #     return self.forecaster.get_tags()

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Sembra ci sia un errore su come fh e' utilizzato in sktime.
    # In teoria, si dovrebbe poter scrivere fh=[1,2,3,4,5], E ANCHE fh=[1,5]
    # Ma nei due casi, y[1] e y[5]  risultano avere valori diversi!
    # Per ovviare al problema, si usa fh=[1,2,3,4,5] e solo DOPO si filtrano
    # i risultati
    #

    def _fit(self, y, X=None, fh=None):
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self

    def _predict(self, fh: ForecastingHorizon, X=None, y=None):
        # convert fh into relative
        fhr = fh.to_relative(self.cutoff).to_numpy()
        # n of slots to predict
        n = fhr[-1]
        # fh = [1,2,3,....,n]
        fhp = ForecastingHorizon(np.arange(1, n+1), is_relative=True)
        # clip of X with datetime starting from cutoff+1 (clip is an array[bool])
        clip = X.index > self.cutoff[0]
        Xp = X
        if not clip[0]: Xp = Xp[clip]
        if len(Xp) > n: Xp = Xp[:n]
        # prediction
        y_pred = self.forecaster.predict(fh=fhp, X=Xp)
        # selection of the required results: fh==1 -> y_pred[0]
        return y_pred.iloc[fhr-1]

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

