from ..base import BaseForecaster

# ForecastingRandomizedSearchCV
#         forecaster,
#         cv,
#         param_distributions,
#         n_iter=10,
#         scoring=None,
#         strategy="refit",
#         refit=True,
#         verbose=0,
#         return_n_best_forecasters=1,
#         random_state=None,
#         backend="loky",
#         update_behaviour="full_refit",
#         error_score=np.nan,
#         tune_by_instance=False,
#         tune_by_variable=False,
#         backend_params=None,
#         n_jobs="deprecated",
#
#
# ForecastingSkoptSearchCV
#         forecaster,
#         cv: BaseSplitter,
#         param_distributions: dict | list[dict],
#         n_iter: int = 10,
#         n_points: int | None = 1,
#         random_state: int | None = None,
#         scoring: list[BaseMetric] | None = None,
#         optimizer_kwargs: dict | None = None,
#         strategy: str | None = "refit",
#         refit: bool = True,
#         verbose: int = 0,
#         return_n_best_forecasters: int = 1,
#         backend: str = "loky",
#         update_behaviour: str = "full_refit",
#         error_score=np.nan,
#         tune_by_instance=False,
#         tune_by_variable=False,
#         backend_params=None,
#         n_jobs="deprecated",
#
#
# ForecastingGridSearchCV
#         forecaster,
#         cv,
#         param_grid,
#         scoring=None,
#         strategy="refit",
#         refit=True,
#         verbose=0,
#         return_n_best_forecasters=1,
#         backend="loky",
#         update_behaviour="full_refit",
#         error_score=np.nan,
#         tune_by_instance=False,
#         tune_by_variable=False,
#         backend_params=None,
#         n_jobs="deprecated",
#
#
# ForecastingOptunaSearchCV
#         forecaster,
#         cv,
#         param_grid,
#         scoring=None,
#         strategy="refit",
#         refit=True,
#         verbose=0,
#         return_n_best_forecasters=1,
#         backend="loky",
#         update_behaviour="full_refit",
#         error_score=np.nan,
#         n_evals=100,
#         sampler=None,

class ForecastingSearchCV(BaseForecaster):
    pass
