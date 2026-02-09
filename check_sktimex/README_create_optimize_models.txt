Introduction
------------

    There are different ways to create an instance of a Python class

        instance = create_from(dict)

        clazz = import_from("qualified_name")
        instance = clazz(**dict)

    For the time series, it is possible to use

        instance = sktimex.forecasting.create_forecaster(name, dict)

    automatically wraps classes in package:

        sklearn, sklearnx, lightgbm, catboost, xgboost


Model search
------------

    Some "sktime" models have an integrated automatic search mechanism.

        AutoETS
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ets.AutoETS.html
        AutoREG
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.auto_reg.AutoREG.html
        AutoTS
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.autots.AutoTS.html
        StatsForecastAutoARIMA
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastAutoARIMA.html
        StatsForecastAutoCES
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastAutoCES.html
        StatsForecastAutoETS
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastAutoETS.html
        StatsForecastAutoTBATS
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastAutoTBATS.html
        StatsForecastAutoTheta
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.statsforecast.StatsForecastAutoTheta.html

    For the other models, "sktime" offer the following wrappers

        ForecastingGridSearchCV
            Perform grid-search cross-validation to find optimal model parameters.
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.ForecastingGridSearchCV.html
        ForecastingRandomizedSearchCV
            The forecaster is fit on the initial window and then temporal cross-validation is used to find the optimal parameter
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.ForecastingRandomizedSearchCV.html

        ForecastingOptCV
            Tune an sktime forecaster via any optimizer in the 'hyperactive' toolbox.
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.ForecastingOptCV.html
        ForecastingOptunaSearchCV
            Perform 'Optuna' search cross-validation to find optimal model hyperparameters.
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.ForecastingOptunaSearchCV.html
        ForecastingSkoptSearchCV
            Bayesian search over hyperparameters for a forecaster using 'scikit-optimize''. [experimental]
            https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.ForecastingSkoptSearchCV.html


    It is possible to use an "automatic wrapping" with a model search algorithm with the following
    trick:

        0) to use ONLY list of parameter values, NOT search on continuous values
        1) a parameter starting with "*" specify that it is required a model search
        2) some "predefined" parameters are used to configure the search algorithm

            "*<param>": [<value1>, ... <valuen>]
            "search_method": "grid" | "random" | "opt" | "optuna" | "skopt"
            "search_<param>": parameter <param> (removing "search_") is passed to the algorithm

    Note: the best parameters MUST BE AUTOMATICALLY SAVED

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

    to see 'README_cv.txt'



"sktime" model
--------------

    "skt.ARDL": {
        "class": "sktime.forecasting.ardl.ARDL",
        "lags": 24
    },


"sklearn" with explicit wrapper
-------------------------------

    "skl.CatBoostRegressor": {
        "class": "sktimex.forecasting.sklearn.ScikitLearnForecaster",
        "estimator": {
            "class": "catboost.core.CatBoostRegressor",
            "silent": true
        },
        "window_length": 24,
        "prediction_length": 6
    },


"sklearn" with automatic wrapper
--------------------------------

    "skl.CatBoostRegressor": {
        "class": "catboost.core.CatBoostRegressor",
        "silent": true
        "window_length": 24,
        "prediction_length": 6
    },

    "skl.CatBoostRegressor": {
        "class": "catboost.core.CatBoostRegressor",
        "silent": true
        "lags": 24,
        "tlags": 6
    },


