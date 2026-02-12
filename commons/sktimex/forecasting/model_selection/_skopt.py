import warnings
from typing import Optional

import skopt.space
from sktime.forecasting.model_selection import ForecastingSkoptSearchCV as Sktime_ForecastingSkoptSearchCV
from stdlib.qname import create_from

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return x


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

            error_score="nan",
            tune_by_instance=False,
            tune_by_variable=False,
            random_state: int | None = None,

            backend: str = "loky",
            backend_params=None,
            verbose: int = 0,
    ):
        # TRICK: force [1,2] to be converted into Categorical([1,2])
        import skopt.space.space as skopt_space
        skopt_space._check_dimension_old = skopt_space._check_dimension

        assert param_grid is None or param_distributions is None, \
            "Only one of 'param_grid' or 'param_distributions' can be not None"

        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
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
            error_score=safe_float(error_score),
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,

            backend=backend,
            backend_params=backend_params,
        )
        self.param_grid = param_grid
        self._forecaster_override = forecaster
        self._cv_override = cv
        self._param_distributions_override = param_distributions
        self._error_score_override = error_score
        return

    # def _create_optimizer(self, params_space: dict):
    #     # reimplement the method to convert 'param_grid' in a dictionary
    #     # of Categorical dimensions
    #     from skopt.optimizer import Optimizer
    #     from skopt.utils import dimensions_aslist
    #     from skopt.space import Categorical
    #
    #     if self.param_grid is not None:
    #         params_space =  {
    #             k: Categorical(params_space[k], name=k)
    #             for k in params_space
    #         }
    #
    #     kwargs = self.optimizer_kwargs_.copy()
    #     # convert params space to a list ordered by the key name
    #     kwargs["dimensions"] = dimensions_aslist(params_space)
    #     dimensions_name = sorted(params_space.keys())
    #     optimizer = Optimizer(**kwargs)
    #     # set the name of the dimensions if not set
    #     for i in range(len(optimizer.space.dimensions)):
    #         if optimizer.space.dimensions[i].name is not None:
    #             continue
    #         optimizer.space.dimensions[i].name = dimensions_name[i]
    #
    #     return optimizer

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
