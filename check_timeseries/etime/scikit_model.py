from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from sktime.forecasting.compose import make_reduction

from stdlib import import_from, dict_del, kwval


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[None, int, list[int], np.ndarray, ForecastingHorizon]


# ---------------------------------------------------------------------------
# ScikitForecastRegressor
# ---------------------------------------------------------------------------

class ScikitForecastRegressor(BaseForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 **kwargs):
        super().__init__()

        model_class = import_from(class_name)

        # extract the top namespace
        p = class_name.find('.')
        ns = class_name[:p]
        if ns in SCIKIT_NAMESPACES:
            window_length = kwval(kwargs, 'window_length', 5)
            strategy = kwval(kwargs, 'strategy', 'recursive')

            kwargs = dict_del(kwargs, ['window_length', 'strategy'])
            # create the regressor
            regressor = model_class(**kwargs)
            # create the forecaster
            self.forecaster = make_reduction(regressor, window_length=window_length, strategy=strategy)
        elif ns in SKTIME_NAMESPACES:
            # create the forecaster
            self.forecaster = model_class(**kwargs)
        else:
            raise ValueError(f"Unsupported class_name '{class_name}'")
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

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)
    #                       == fit(y, X[:y]
    #                          predict(fh, X[y:]
    #

    def fit(self, y, X=None, fh: FH_TYPES = None):
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self

    def predict(self,
                fh: FH_TYPES = None,
                X: Optional[pd.DataFrame] = None,
                y: Union[None, pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        fh = self._resolve_fh(y, X, fh)

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        Xp, yh, Xh = self._prepare_predict(X, y)

        # retrain if yh (and Xh) are available
        if yh is not None:
            self.fit(y=yh, X=Xh)

        n = fh[-1]
        fhp = ForecastingHorizon(np.arange(1, n + 1))
        y_pred = self.forecaster.predict(fh=fhp, X=Xp)

        assert isinstance(y_pred, pd.DataFrame)
        return y_pred.iloc[fh-1]

    def _resolve_fh(self, y, X, fh: FH_TYPES) -> ForecastingHorizon:
        # (_, _, fh)        -> fh
        # (X, None, None)   -> |X|
        # (None, y, None)   -> error
        # (X, y, None)      -> |X| - |y|

        if fh is not None:
            cutoff = self.cutoff if y is None else y.index[-1]
            fh = fh if isinstance(fh, ForecastingHorizon) else ForecastingHorizon(fh)
            return fh.to_relative(cutoff)
        if y is None:
            n = len(X)
            return ForecastingHorizon(np.arange(1, n+1))
        else:
            n = len(X) - len(y)
            return ForecastingHorizon(np.arange(1, n+1))
    # end

    def _prepare_predict(self, X, y):

        if y is None:
            return X, None, None
        if X is None:
            return None, y, None

        n = len(y)
        Xh = X[:n]
        yh = y
        Xp = X[n:]
        return Xp, yh, Xh
    # end

    # def _update_fit(self, y, X):
    #     if X is None or y is None or len(X) == len(y):
    #         return
    #
    #     y_upd = y[y.index > self.cutoff]
    #     X_upd = X[y.index]
    #     self.forecaster.update(y=y_upd, X=X_upd)
    # # end

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
    # score
    # -----------------------------------------------------------------------

    def score(self, y_true, X=None, fh=None) -> dict[str, float]:
        y_pred = self.predict(fh=fh, X=X)
        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def set_scores(self, scores):
        self._scores = scores

    def get_scores(self):
        return self._scores

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

