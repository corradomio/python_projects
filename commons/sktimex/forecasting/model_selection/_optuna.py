import optuna.samplers
from optuna.distributions import CategoricalDistribution
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV as Sktime_ForecastingOptunaSearchCV
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# https://optuna.readthedocs.io/en/stable/reference/samplers/index.html


def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def to_optuna_distributions(param_grid: dict):
    return {
        name: CategoricalDistribution(choices)

        for name,choices in param_grid.items()
    }


# ---------------------------------------------------------------------------
# ForecastingOptunaSearchCV
# ---------------------------------------------------------------------------
# Note:
#   'param_grid' must be a dictionary of 'optuna samplers'

class ForecastingOptunaSearchCV(Sktime_ForecastingOptunaSearchCV):
    """
    Added support to create the class using a dict/JSON object
    """

    def __init__(
            self,
            forecaster,
            cv,
            param_grid,

            scoring=None,
            strategy="refit",
            refit=True,
            update_behaviour="full_refit",

            n_iter=-1,              # alternative to n_evals
            return_n_best_forecasters=1,
            error_score="nan",

            n_evals=100,
            sampler=None,

            backend="loky",
            backend_params=None,    # for compatibility
            verbose=0,
    ):
        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
            param_grid=to_optuna_distributions(param_grid),
            scoring=scoring,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=safe_float(error_score),
            n_evals=n_iter if n_iter > 0 else n_evals,
            sampler=sampler
        )
        self.n_iter = n_iter
        self.backend_params = backend_params
        self._forecaster_override=forecaster
        self._cv_override=cv
        self._param_grid_override=param_grid
        self._error_score_override = error_score
        return

    def _optuna_samplers(self, param_grid):
        return param_grid

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # OVERRIDE the parameters passed as configuration and NOT as Python objects
        params['forecaster'] = self._forecaster_override
        params['cv'] = self._cv_override
        params['param_grid'] = self._param_grid_override
        params["error_score"] = self._error_score_override
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self
