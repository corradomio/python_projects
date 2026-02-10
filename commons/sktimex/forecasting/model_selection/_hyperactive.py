from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ForecastingOptCV as Sktime_ForecastingOptCV
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# ForecastingHyperactiveSearchCV
# ---------------------------------------------------------------------------
# Hyperactive provides 31 optimization algorithms across 3 backends (GFO,
# Optuna, scikit-learn), accessible through a unified experiment-based interface.
# The library separates optimization problems from algorithms, enabling you
# to swap optimizers without changing your experiment code.
#
# Designed for hyperparameter tuning, model selection, and black-box optimization.
# Native integrations with scikit-learn, sktime, skpro, and PyTorch allow tuning
# ML models with minimal setup. Define your objective, specify a search space,
# and run.
#
# GFO: Gradient Free Optimizers
#   https://github.com/SimonBlanke/Gradient-Free-Optimizers
#
# Optuna
#   https://optuna.readthedocs.io/en/stable/#
#
# scikit-learn
#   https://scikit-learn.org/stable/
#
# Hyperactive
#   https://github.com/hyperactive-project/Hyperactive
#


def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return x


# ---------------------------------------------------------------------------
# ForecastingHyperactiveSearchCV
# ---------------------------------------------------------------------------

class ForecastingHyperactiveSearchCV(Sktime_ForecastingOptCV):
    """
    Added support to create the class using a dict/JSON object
    """

    def __init__(
            self,
            forecaster: str|dict,
            cv: str | dict,
            # param_grid: dict,
            pred_len: int,

            optimizer: str | dict,

            scoring=None,
            strategy="refit",
            refit=True,
            update_behaviour="full_refit",

            n_iter=-1,              # for compatibility/ignored

            error_score="nan",
            cv_X=None,

            backend=None,
            backend_params=None,
            verbose=0               # for compatibility/ignored
    ):
        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        optimizer_instance = create_from(optimizer)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
            optimizer=optimizer_instance,
            scoring=scoring,
            strategy=strategy,
            refit=refit,

            cv_X=cv_X,
            update_behaviour=update_behaviour,
            error_score=safe_float(error_score),

            backend=backend,
            backend_params=backend_params,
        )
        self.n_iter = n_iter
        self.pred_len = pred_len
        self.verbose = verbose
        self._forecaster_override = forecaster
        self._optimizer_override = optimizer
        self._cv_override = cv
        self._error_score_override = error_score
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len + 1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # OVERRIDE the parameters passed as configuration and NOT as Python objects
        params["forecaster"] = self._forecaster_override
        params["optimizer"] = self._optimizer_override
        params["cv"] = self._cv_override
        params["error_score"] = self._error_score_override
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self
