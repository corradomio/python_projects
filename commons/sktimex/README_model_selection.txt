ForecastingGridSearchCV
ForecastingRandomizedSearchCV
ForecastingOptCV
    hyperactive
ForecastingSkoptSearchCV
    scikit-optimize
ForecastingOptunaSearchCV
    optuna


cv
--

    BaseWindowSplitter(BaseSplitter) (sktime.split.base._base_windowsplitter)
        ExpandingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingwindow)
        SlidingWindowSplitter(BaseWindowSplitter) (sktime.split.slidingwindow)
        ExpandingSlidingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingslidingwindow)


score
-----

    BaseMetric (sktime.benchmarking.base)
        PairwiseMetric(BaseMetric) (sktime.benchmarking.metrics)
        AggregateMetric(BaseMetric) (sktime.benchmarking.metrics)



Parameters
----------

# ForecastingGridSearchCV
#         forecaster,
#         cv,
#         param_grid,                   ~= param_distributions
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
# ForecastingRandomizedSearchCV
#         forecaster,
#         cv,
#         param_distributions,          ~= param_grid
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



Attributes
----------

ForecastingGridSearchCV
    Attributes:

        best_index_: int
        best_score_: float
            Score of the best model
        best_params_ dict
            Best parameter values across the parameter grid
        best_forecaster_ estimator
            Fitted estimator with the best parameters
        cv_results_ dict
            Results from grid search cross validation
        n_splits_: int
            Number of splits in the data for cross validation
        refit_time_ float
            Time (seconds) to refit the best forecaster
        scorer_ function
            Function used to score model
        n_best_forecasters_: list of tuples (“rank”, <forecaster>)
            The “rank” is in relation to best_forecaster_
        n_best_scores_: list of float
            The scores of n_best_forecasters_ sorted from best to worst score of forecasters
        forecasters_ pd.DataFrame

            DataFrame with all fitted forecasters and their parameters. Only present if
            tune_by_instance=True or tune_by_variable=True, and at least one of the two is
            applicable. In this case, the other attributes are not present in self, only
            in the fields of forecasters_.



ForecastingRandomizedSearchCV
    Attributes:

        best_index_: int
        best_score_: float
            Score of the best model
        best_params_ dict
            Best parameter values across the parameter grid
        best_forecaster_ estimator
            Fitted estimator with the best parameters
        cv_results_ dict
            Results from grid search cross validation
        n_best_forecasters_: list of tuples (“rank”, <forecaster>)
            The “rank” is in relation to best_forecaster_
        n_best_scores_: list of float
            The scores of n_best_forecasters_ sorted from best to worst score of forecasters
        forecasters_ pd.DataFrame

            DataFrame with all fitted forecasters and their parameters. Only present if
            tune_by_instance=True or tune_by_variable=True, and at least one of the two is
            applicable. In this case, the other attributes are not present in self, only
            in the fields of forecasters_.



ForecastingSkoptSearchCV
    Attributes:

        best_index_ int
        best_score_: float
            Score of the best model
        best_params_ dict
            Best parameter values across the parameter grid
        best_forecaster_ estimator
            Fitted estimator with the best parameters
        cv_results_ dict
            Results from grid search cross validation
        n_best_forecasters_: list of tuples (“rank”, <forecaster>)
            The “rank” is in relation to best_forecaster_
        n_best_scores_: list of float
            The scores of n_best_forecasters_ sorted from best to worst score of forecasters
        forecasters_ pd.DataFrame

            DataFrame with all fitted forecasters and their parameters. Only present if
            tune_by_instance=True or tune_by_variable=True, and at least one of the two is
            applicable. In this case, the other attributes are not present in self, only
            in the fields of forecasters_.



ForecastingHiperactive/ForecastingOptCV
    Attributes:

        cutoff
            Cut-off = “present time” state of forecaster.
        fh
            Forecasting horizon that was passed.
        is_fitted
            Whether fit has been called.


ForecastingOptunaSearchCV

    best_index_ int
    best_score_: float
        Score of the best model
    best_params_ dict
        Best parameter values across the parameter grid
    best_forecaster_ estimator
        Fitted estimator with the best parameters
    cv_results_ dict
        Results from grid search cross validation
    n_best_forecasters_: list of tuples (“rank”, <forecaster>)
        The “rank” is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst score of forecasters

