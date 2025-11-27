sktime.forecasting.model_selection.[
    _base,
    _gridsearch,
    _randomsearch,
    _hyperactive,
    _optuna
]
-----------------------------------------------------------------------------


ForecastingGridSearchCV
    Perform grid-search cross-validation to find optimal model parameters.
ForecastingRandomizedSearchCV
    The forecaster is fit on the initial window and then temporal cross-validation is used to find the optimal parameter

ForecastingOptCV
    Tune an sktime forecaster via any optimizer in the 'hyperactive' toolbox.
ForecastingOptunaSearchCV
    Perform 'Optuna' search cross-validation to find optimal model hyperparameters.
ForecastingSkoptSearchCV
    Bayesian search over hyperparameters for a forecaster using 'scikit-optimize''. [experimental]

-----------------------------------------------------------------------------

ForecastingGridSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._gridsearch)
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
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
        n_jobs="deprecated",


ForecastingRandomizedSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._randomsearch)
        self,
        forecaster,
        cv,
        param_distributions,
        n_iter=10,
        scoring=None,
        strategy="refit",
        refit=True,
        verbose=0,
        return_n_best_forecasters=1,
        random_state=None,
        backend="loky",
        update_behaviour="full_refit",
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
        n_jobs="deprecated",


ForecastingOptCV
        self,
        forecaster,
        optimizer,
        cv,
        strategy="refit",
        update_behaviour="full_refit",
        scoring=None,
        refit=True,
        error_score=np.nan,
        cv_X=None,
        backend=None,
        backend_params=None,


ForecastingSkoptSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._skopt)
        self,
        forecaster,
        cv: BaseSplitter,
        param_distributions: dict | list[dict],
        n_iter: int = 10,
        n_points: int | None = 1,
        random_state: int | None = None,
        scoring: list[BaseMetric] | None = None,
        optimizer_kwargs: dict | None = None,
        strategy: str | None = "refit",
        refit: bool = True,
        verbose: int = 0,
        return_n_best_forecasters: int = 1,
        backend: str = "loky",
        update_behaviour: str = "full_refit",
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
        n_jobs="deprecated",

ForecastingOptunaSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._optuna)
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
        error_score=np.nan,
        n_evals=100,
        sampler=None,

-----------------------------------------------------------------------------
cv: Cross Validatione

BaseSplitter
    CutoffSplitter(BaseSplitter) (sktime.split.cutoff)
    CutoffFhSplitter(BaseSplitter) (sktime.split.cutoff)
    ExpandingGreedySplitter(BaseSplitter) (sktime.split.expandinggreedy)
    ExpandingCutoffSplitter(BaseSplitter) (sktime.split.expandingcutoff)
    SlidingGreedySplitter(BaseSplitter) (sktime.split.slidinggreedy)
    SameLocSplitter(BaseSplitter) (sktime.split.sameloc)
    TemporalTrainTestSplitter(BaseSplitter) (sktime.split.temporal_train_test_split)
    SingleWindowSplitter(BaseSplitter) (sktime.split.singlewindow)
    InstanceSplitter(BaseSplitter) (sktime.split.instance)
    TestPlusTrainSplitter(BaseSplitter) (sktime.split.testplustrain)
    ForecastingHorizonSplitter(BaseSplitter) (sktime.split.fh)
    BaseWindowSplitter(BaseSplitter) (sktime.split.base._base_windowsplitter)
        ExpandingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingwindow)
        SlidingWindowSplitter(BaseWindowSplitter) (sktime.split.slidingwindow)
        ExpandingSlidingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingslidingwindow)
    Repeat(BaseSplitter) (sktime.split.compose._repeat)
    MySplitter(BaseSplitter) (extension_templates.split)
