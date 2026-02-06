from typing import Optional

from sktime.forecasting.model_selection import ForecastingSkoptSearchCV as Sktime_ForecastingSkoptSearchCV
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# ForecastingRandomizedSearchCV
# ---------------------------------------------------------------------------

class ForecastingSkoptSearchCV(Sktime_ForecastingSkoptSearchCV):
    """
    Added support to create the class using a dict/JSON object
    """

    def __init__(
            self,
            forecaster: str | dict,
            cv: str | dict,
            param_grid: Optional[dict] = None,
            param_distributions: None | dict | list[dict] = None,

            scoring=None,
            strategy: str | None = "refit",
            refit: bool = True,
            update_behaviour: str = "full_refit",

            n_iter: int = 10,
            n_points: int | None = 1,
            optimizer_kwargs: dict | None = None,

            return_n_best_forecasters: int = 1,
            random_state: int | None = None,
            error_score="nan",
            tune_by_instance=False,
            tune_by_variable=False,

            backend: str = "loky",
            backend_params=None,
            verbose: int = 0,
    ):
        assert param_grid is None or param_distributions is None, \
            "Only one of 'param_grid' or 'param_distributions' can be not None"

        super().__init__(
            forecaster=create_from(forecaster),
            cv=create_from(cv),
            param_distributions=param_distributions if param_grid is None else param_grid,
            n_iter=n_iter,
            n_points=n_points,
            scoring=scoring,
            optimizer_kwargs=optimizer_kwargs,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            random_state=random_state,
            update_behaviour=update_behaviour,
            error_score=error_score,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,

            backend=backend,
            backend_params=backend_params,
        )
        self._forecaster_override = forecaster
        self._cv_override = cv
        self._param_distributions_override = param_distributions
        self.param_grid = param_grid
        self._error_score_override = error_score
        return

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # OVERRIDE the parameters passed as configuration and NOT as Python objects
        params['forecaster'] = self._forecaster_override
        params['cv'] = self._cv_override
        params['param_distributions'] = self._param_distributions_override
        params["error_score"] = self._error_score_override
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self
