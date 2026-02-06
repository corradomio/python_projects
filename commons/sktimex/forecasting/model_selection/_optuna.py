import optuna.samplers
from optuna.distributions import CategoricalDistribution
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV as Sktime_ForecastingOptunaSearchCV
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# https://optuna.readthedocs.io/en/stable/reference/samplers/index.html


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

            return_n_best_forecasters=1,
            error_score="nan",
            n_evals=100,
            sampler=None,

            backend="loky",
            backend_params=None,    # for compatibility
            verbose=0,
    ):

        super().__init__(
            forecaster=create_from(forecaster),
            cv=create_from(cv),
            param_grid=to_optuna_distributions(param_grid),
            scoring=scoring,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=float(error_score),
            n_evals=n_evals,
            sampler=sampler
        )
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
