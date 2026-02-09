<<<<<<< Updated upstream

from sktime.forecasting.model_selection._base import BaseGridSearch
=======
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._base import BaseGridSearch
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter
from sktime.utils.parallel import parallelize
from sktime.utils.validation.forecasting import check_scoring
from stdlib.qname import create_from
>>>>>>> Stashed changes


# ---------------------------------------------------------------------------
# ForecastingGeneticAlgorithmSearchCV
# ---------------------------------------------------------------------------

<<<<<<< Updated upstream

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
=======
class ForecastingGeneticAlgorithmSearchCV(BaseGridSearch):

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
        super().__init__(
            forecaster=create_from(forecaster),
            cv=create_from(cv),
            scoring=scoring,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            update_behaviour=update_behaviour,
            error_score=float(error_score),
            backend=backend,
            backend_params=backend_params,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
>>>>>>> Stashed changes
        )
        self.param_grid = param_grid
        self._forecaster_override=forecaster
        self._cv_override=cv
        self._error_score_override = error_score
<<<<<<< Updated upstream
=======
        return
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream


=======
>>>>>>> Stashed changes
