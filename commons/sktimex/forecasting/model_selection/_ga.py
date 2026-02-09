
from sktime.forecasting.model_selection._base import BaseGridSearch


# ---------------------------------------------------------------------------
# ForecastingGeneticAlgorithmSearchCV
# ---------------------------------------------------------------------------


class ForecastingGeneticAlgorithmSearchCV(BaseGridSearch):

    def __init__(
        self,
        forecaster,
        cv,
        param_grid,
        scoring=None,
        strategy="refit",
        refit=True,
        verbose=0,
        return_n_best_forecasters=1,
        backend="loky",
        update_behaviour="full_refit",
        error_score="nan",
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            refit=refit,
            cv=cv,
            strategy=strategy,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=float(error_score),
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
            backend_params=backend_params,
        )
        self.param_grid = param_grid
        self._forecaster_override=forecaster
        self._cv_override=cv
        self._error_score_override = error_score

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


