
-----------------------------------------------------------------------------

class ForecastingGridSearchCV(BaseGridSearch):
    """Perform grid-search cross-validation to find optimal model parameters.

    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter.

    Grid-search cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter grid to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same forecaster with different hyper-parameters.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
    cv : cross-validation generator or an iterable
        e.g. SlidingWindowSplitter()
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
    param_grid : dict or list of dictionaries
        Model tuning parameters of the forecaster to evaluate

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose: int, optional (default=0)
    return_n_best_forecasters : int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_.
        Set return_n_best_forecasters to -1 to return all forecasters.

    error_score : numeric value or the str 'raise', optional (default=np.nan)
        The test score returned when a forecaster fails to be fitted.
    return_train_score : bool, optional (default=False)

    backend : {"dask", "loky", "multiprocessing", "threading","ray"}, by default "loky".
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_splits_: int
        Number of splits in the data for cross validation
    refit_time_ : float
        Time (seconds) to refit the best forecaster
    scorer_ : function
        Function used to score model
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_shampoo_sales()
    >>> fh = [1,2,3]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = NaiveForecaster()
    >>> param_grid = {"strategy" : ["last", "mean", "drift"]}
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=forecaster,
    ...     param_grid=param_grid,
    ...     cv=cv)
    >>> gscv.fit(y)
    ForecastingGridSearchCV(...)
    >>> y_pred = gscv.predict(fh)

        Advanced model meta-tuning (model selection) with multiple forecasters
        together with hyper-parametertuning at same time using sklearn notation:

    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.transformations.series.impute import Imputer
    >>> y = load_shampoo_sales()
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("imputer", Imputer()),
    ...     ("forecaster", NaiveForecaster())])
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1,2,3])
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=pipe,
    ...     param_grid=[{
    ...         "forecaster": [NaiveForecaster(sp=12)],
    ...         "forecaster__strategy": ["drift", "last", "mean"],
    ...     },
    ...     {
    ...         "imputer__method": ["mean", "drift"],
    ...         "forecaster": [ThetaForecaster(sp=12)],
    ...     },
    ...     {
    ...         "imputer__method": ["mean", "median"],
    ...         "forecaster": [ExponentialSmoothing(sp=12)],
    ...         "forecaster__trend": ["add", "mul"],
    ...     },
    ...     ],
    ...     cv=cv,
    ... )  # doctest: +SKIP
    >>> gscv.fit(y)  # doctest: +SKIP
    ForecastingGridSearchCV(...)
    >>> y_pred = gscv.predict(fh=[1,2,3])  # doctest: +SKIP
    """

-----------------------------------------------------------------------------

class ForecastingRandomizedSearchCV(BaseGridSearch):
    """Perform randomized-search cross-validation to find optimal model parameters.

    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter

    Randomized cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter distributions to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same forecaster with different hyper-parameters.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
    cv : cross-validation generator or an iterable
        e.g. SlidingWindowSplitter()
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    update_behaviour: str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
    param_distributions : dict or list of dicts
        Dictionary with parameters names (``str``) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose : int, optional (default=0)
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_.
        Set return_n_best_forecasters to -1 to return all forecasters.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.
    """

-----------------------------------------------------------------------------

class ForecastingSkoptSearchCV(BaseGridSearch):
    """Bayesian search over hyperparameters for a forecaster.

    Experimental: This feature is under development and interface may likely to change.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
    cv : cross-validation generator or an iterable
        Splitter used for generating validation folds.
        e.g. SlidingWindowSplitter()
    param_distributions : dict or a list of dict/tuple. See below for details.
        1. If dict, a dictionary that represents the search space over the parameters of
        the provided estimator. The keys are parameter names (strings), and the values
        follows the following format. A list to store categorical parameters and a
        tuple for integer and real parameters with the following format
        (int/float, int/float, "prior") e.g (1e-6, 1e-1, "log-uniform").
        2. If a list of dict, each dictionary corresponds to a parameter space,
        following the same structure described in case 1 above. the search will be
        performed sequentially for each parameter space, with the number of samples
        set to n_iter.
        3. If a list of tuple, tuple must contain (dict, int) where the int refers to
        n_iter for that search space. dict must follow the same structure as in case 1.
        This is useful if you want to perform a search with different number of
        iterations for each parameter space.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution. Consider increasing n_points
        if you want to try more parameter settings in parallel.
    n_points : int, default=1
        Number of parameter settings to sample in parallel.
        If this does not align with n_iter, the last iteration will sample less points

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    optimizer_kwargs: dict, optional
        Arguments passed to Optimizer to control the behaviour of the bayesian search.
        For example, {'base_estimator': 'RF'} would use a Random Forest surrogate
        instead of the default Gaussian Process. Please refer to the ``skopt.Optimizer``
        documentation for more information.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    update_behaviour: str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose : int, optional (default=0)
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_.
        Set return_n_best_forecasters to -1 to return all forecasters.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingSkoptSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sktime.forecasting.compose import make_reduction
    >>> y = load_shampoo_sales()
    >>> fh = [1,2,3,4]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = make_reduction(GradientBoostingRegressor(random_state=10))
    >>> param_distributions = {
    ...     "estimator__learning_rate" : (1e-4, 1e-1, "log-uniform"),
    ...     "window_length" : (1, 10, "uniform"),
    ...     "estimator__criterion" : ["friedman_mse", "squared_error"]}
    >>> sscv = ForecastingSkoptSearchCV(
    ...     forecaster=forecaster,
    ...     param_distributions=param_distributions,
    ...     cv=cv,
    ...     n_iter=5,
    ...     random_state=10)  # doctest: +SKIP
    >>> sscv.fit(y)  # doctest: +SKIP
    ForecastingSkoptSearchCV(...)
    >>> y_pred = sscv.predict(fh)  # doctest: +SKIP
    """

-----------------------------------------------------------------------------

class ForecastingOptunaSearchCV(BaseGridSearch):
    """Perform Optuna search cross-validation to find optimal model hyperparameters.

    Experimental: This feature is under development and interfaces may change.

    In ``fit``, this estimator uses the ``optuna`` base search algorithm
    applied to the ``sktime`` ``evaluate`` benchmarking output.

    ``param_grid`` is used to parametrize the search space, over parameters of
    the passed ``forecaster``, via ``set_params``.

    The remaining parameters are passed directly to ``evaluate``, to obtain
    the primary optimization outcome as the aggregate ``scoring`` metric specified
    on the evaluation schema.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
    cv : cross-validation generator or an iterable
        Splitter used for generating validation folds.
        e.g. ExpandingWindowSplitter()
    param_grid : dict of optuna samplers
        Dictionary with parameters names as keys and lists of parameter distributions
        from which to sample parameter values.
        e.g. {"forecaster": optuna.distributions.CategoricalDistribution(
        (STLForecaster(), ThetaForecaster())}
    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
        Valid strings are valid registry.craft specs, which include
        string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
        and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    return_n_best_forecasters : int, default=1
        Number of best forecasters to return.
    backend : str, default="loky"
        Backend to use when running the fit.
    update_behaviour : str, default="full_refit"
        Determines how to update the forecaster during fitting.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    n_evals : int, default=100
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    sampler : Optuna sampler, optional (default=None)
        e.g. optuna.samplers.TPESampler(seed=42)

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters

    Examples
    --------
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingOptunaSearchCV,
    ...     )
    >>> from sktime.datasets import load_shampoo_sales
    >>> import warnings
    >>> warnings.simplefilter(action="ignore", category=FutureWarning)
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.split import temporal_train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler, RobustScaler
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.transformations.series.detrend import Deseasonalizer, Detrender
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import STLForecaster, TrendForecaster
    >>> import optuna
    >>> from  optuna.distributions import CategoricalDistribution

    >>> y = load_shampoo_sales()
    >>> y_train, y_test = temporal_train_test_split(y=y, test_size=6)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
    ...         cutoff=y_train.index[-1]
    ...     )
    >>> cv = ExpandingWindowSplitter(fh=fh, initial_window=24, step_length=1)
    >>> forecaster = TransformedTargetForecaster(
    ...     steps=[
    ...             ("detrender", Detrender()),
    ...             ("scaler", RobustScaler()),
    ...             ("minmax2", MinMaxScaler((1, 10))),
    ...             ("forecaster", NaiveForecaster()),
    ...         ]
    ...     )
    >>> param_grid = {
    ...     "scaler__with_scaling": CategoricalDistribution(
    ...             (True, False)
    ...         ),
    ...     "forecaster": CategoricalDistribution(
    ...             (NaiveForecaster(), TrendForecaster())
    ...         ),
    ...     }
    >>> gscv = ForecastingOptunaSearchCV(
    ...         forecaster=forecaster,
    ...         param_grid=param_grid,
    ...         cv=cv,
    ...         n_evals=10,
    ...     )
    >>> gscv.fit(y)
    ForecastingOptunaSearchCV(...)
    >>> print(f"{gscv.best_params_=}")  # doctest: +SKIP
    """

-----------------------------------------------------------------------------


BaseSplitter(BaseObject) (sktime.split.base._base_splitter)
    CutoffSplitter(BaseSplitter) (sktime.split.cutoff)
    CutoffFhSplitter(BaseSplitter) (sktime.split.cutoff)
    ExpandingCutoffSplitter(BaseSplitter) (sktime.split.expandingcutoff)
    ExpandingGreedySplitter(BaseSplitter) (sktime.split.expandinggreedy)
    SlidingGreedySplitter(BaseSplitter) (sktime.split.slidinggreedy)
    SameLocSplitter(BaseSplitter) (sktime.split.sameloc)
    TemporalTrainTestSplitter(BaseSplitter) (sktime.split.temporal_train_test_split)
    InstanceSplitter(BaseSplitter) (sktime.split.instance)
    SingleWindowSplitter(BaseSplitter) (sktime.split.singlewindow)
    TestPlusTrainSplitter(BaseSplitter) (sktime.split.testplustrain)
    ForecastingHorizonSplitter(BaseSplitter) (sktime.split.fh)
    Repeat(BaseSplitter) (sktime.split.compose._repeat)

    BaseWindowSplitter(BaseSplitter) (sktime.split.base._base_windowsplitter)
        ExpandingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingwindow)
        SlidingWindowSplitter(BaseWindowSplitter) (sktime.split.slidingwindow)
        ExpandingSlidingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingslidingwindow)


    BaseWindowSplitter
        fh: FORECASTING_HORIZON_TYPES,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
        start_with_window: bool,
        max_expanding_window_length: ACCEPTED_WINDOW_LENGTH_TYPES = float("inf"),

    ExpandingWindowSplitter
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,

    SlidingWindowSplitter
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES | None = None,
        start_with_window: bool = True,

    ExpandingSlidingWindowSplitter
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        max_expanding_window_length: ACCEPTED_WINDOW_LENGTH_TYPES = float("inf"),