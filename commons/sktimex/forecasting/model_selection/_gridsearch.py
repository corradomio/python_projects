from sktime.forecasting.model_selection import ForecastingGridSearchCV as Sktime_ForecastingGridSearchCV
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
# ForecastingGridSearchCV
# ---------------------------------------------------------------------------

class ForecastingGridSearchCV(Sktime_ForecastingGridSearchCV):
    """
    Added support to create the class using a dict/JSON object
    """

    def __init__(
            self,
            forecaster: str | dict,
            cv: str | dict,
            param_grid: dict,
            scoring=None,
            strategy="refit",
            refit=True,
            update_behaviour="full_refit",

            return_n_best_forecasters=1,

            error_score="nan",
            tune_by_instance=False,
            tune_by_variable=False,

            backend="loky",
            backend_params=None,
            verbose=0,
    ):
        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
            param_grid=param_grid,
            scoring=scoring,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            update_behaviour=update_behaviour,
            error_score=safe_float(error_score),
            backend=backend,
            backend_params=backend_params,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable
        )
        self._forecaster_override=forecaster
        self._cv_override=cv
        self._error_score_override = error_score
        return

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # OVERRIDE the parameters passed as configuration and NOT as Python objects
        params['forecaster'] = self._forecaster_override
        params['cv'] = self._cv_override
        params["error_score"] = self._error_score_override
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self
