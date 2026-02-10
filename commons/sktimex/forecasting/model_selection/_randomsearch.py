from typing import Optional

from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV as Sktime_ForecastingRandomizedSearchCV
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

class ForecastingRandomizedSearchCV(Sktime_ForecastingRandomizedSearchCV):
    """
    Extends

        'sktime.forecasting.model_selection.ForecastingRandomizedSearchCV'

    supporting the parameter 'param_grid', converted into 'param_distributions'
    where each value in the grid is selected randomly with uniform distribution.
    This permits to configure 'ForecastingRandomizedSearchCV' and 'ForecastingGridSearchCV'
    without to change the parameters.

    Added support to create the class using a dict/JSON object
    """

    def __init__(
            self,
            forecaster: str | dict,
            cv: str | dict,
            param_grid: Optional[dict] =None,
            param_distributions: Optional[dict]=None,

            scoring=None,
            strategy="refit",
            refit=True,
            update_behaviour="full_refit",

            n_iter=10,
            return_n_best_forecasters=1,
            error_score="nan",
            random_state=None,
            tune_by_instance=False,
            tune_by_variable=False,

            backend="loky",
            backend_params=None,
            verbose=0,
    ):
        assert param_grid is None or param_distributions is None, \
            "Only one of 'param_grid' or 'param_distributions' can be not None"

        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
            param_distributions=param_distributions if param_grid is None else param_grid,
            n_iter=n_iter,
            scoring=scoring,
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
