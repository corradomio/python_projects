sktimex.forecasting.ReducerForecaster
-------------------------------------
        estimator="sklearn.linear_model.LinearRegression",
        strategy: Literal["direct", "recursive", "multioutput", "dirrec"] = "recursive",
        window_length=10,
        prediction_length=1,
        scitype="infer",
        transformers=None,
        pooling="local",
        windows_identical=True

    based on sktime   'make_reducer'


sktimex.forecasting.ScikitLearnForecaster
-----------------------------------------
        estimator: Union[str, type, dict] = "sklearn.linear_model.LinearRegression",
        window_length=10,
        prediction_length=1

    simple (first implementation) direct/recursive wrapper on scikit-learn models


sktimex.forecasting.RegressorForecaster
---------------------------------------
        lags: Union[int, list, tuple] = 10,
        tlags: Union[int, list] = 1,
        estimator: Union[str, dict] = "sklearn.linear_model.LinearRegression",
        flatten=True,
        debug=False

    extended version of 'ScikitLearnForecaster' where it is possible to specify
    the y-lags, X-lags and future X-lags to use (Sid/EBTIC approach)
    Note: 'lags' has the form:

        int             == [int, int, 0]        [y-lags, X-lags, 0]
        [int]           == [int, 0, 0]          [y-lags, X-lags, 0]
        [int,int]       == [int, int, 0]        [y-lags, X-lags, 0]
        [int,int,int]                           [y-lags, X-lags, u-lags]

    [X_past  ][y_past]
    [X_future][y_pred]

    u-lags refers to X_future


















_Reducer(_BaseWindowForecaster) (sktime.forecasting.compose._reduce)
 (*)_DirectReducer(_Reducer) (sktime.forecasting.compose._reduce)
        DirectTabularRegressionForecaster(_DirectReducer) (sktime.forecasting.compose._reduce)
        DirectTimeSeriesRegressionForecaster(_DirectReducer) (sktime.forecasting.compose._reduce)
 (*)_MultioutputReducer(_Reducer) (sktime.forecasting.compose._reduce)
        MultioutputTabularRegressionForecaster(_MultioutputReducer) (sktime.forecasting.compose._reduce)
        MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer) (sktime.forecasting.compose._reduce)
    _RecursiveReducer(_Reducer) (sktime.forecasting.compose._reduce)
        RecursiveTabularRegressionForecaster(_RecursiveReducer) (sktime.forecasting.compose._reduce)
        RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer) (sktime.forecasting.compose._reduce)
 (*)_DirRecReducer(_Reducer) (sktime.forecasting.compose._reduce)
        DirRecTabularRegressionForecaster(_DirRecReducer) (sktime.forecasting.compose._reduce)
        DirRecTimeSeriesRegressionForecaster(_DirRecReducer) (sktime.forecasting.compose._reduce)


make_reduction
    DirectTabularRegressionForecaster
    DirectTimeSeriesRegressionForecaster
---
    MultioutputTabularRegressionForecaster
    MultioutputTimeSeriesRegressionForecaster
---
    RecursiveTabularRegressionForecaster
    RecursiveTimeSeriesRegressionForecaster
---
    DirRecTabularRegressionForecaster
    DirRecTimeSeriesRegressionForecaster


--- ??
    DirectReductionForecaster
        """Direct reduction forecaster, incl single-output, multi-output, exogeneous Dir.

        Implements direct reduction, of forecasting to tabular regression.

        For no ``X``, defaults to DirMO (direct multioutput) for ``X_treatment =
        "concurrent"``,
        and simple direct (direct single-output) for ``X_treatment = "shifted"``.

        Direct single-output with concurrent ``X`` behaviour can be configured
        by passing a single-output ``scikit-learn`` compatible transformer.

        Algorithm details:

        In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
            fits ``estimator`` to feature-label pairs as defined as follows.
        if `X_treatment = "concurrent":
            features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
            ``X(t+h)``
            labels = ``y(t+h)`` for ``h`` in the forecasting horizon
            ranging over all ``t`` where the above have been observed (are in the index)
            for each ``h`` in the forecasting horizon (separate estimator fitted per ``h``)
        if `X_treatment = "shifted":
            features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
            ``X(t)``
            labels = ``y(t+h_1)``, ..., ``y(t+h_k)`` for ``h_j`` in the forecasting horizon
            ranging over all ``t`` where the above have been observed (are in the index)
            estimator is fitted as a multi-output estimator (for all ``h_j``
            simultaneously)

        In ``predict``, given possibly exogeneous ``X``, at cutoff time ``c``,
        if `X_treatment = "concurrent":
            applies fitted estimators' predict to
            feature = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
            ``X(c+h)``
            to obtain a prediction for ``y(c+h)``, for each ``h`` in the forecasting horizon
        if `X_treatment = "shifted":
            applies fitted estimator's predict to
            features = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
            ``X(c)``
            to obtain prediction for ``y(c+h_1)``, ..., ``y(c+h_k)`` for ``h_j`` in forec.
            horizon

        Parameters
        ----------
        estimator : sklearn regressor, must be compatible with sklearn interface
            tabular regression algorithm used in reduction algorithm

        window_length : int, optional, default=10
            window length used in the reduction algorithm

        transformers : currently not used

        X_treatment : str, optional, one of "concurrent" (default) or "shifted"
            determines the timestamps of X from which y(t+h) is predicted, for horizon h
            "concurrent": y(t+h) is predicted from lagged y, and X(t+h), for all h in fh
                in particular, if no y-lags are specified, y(t+h) is predicted from X(t)
            "shifted": y(t+h) is predicted from lagged y, and X(t), for all h in fh
                in particular, if no y-lags are specified, y(t+h) is predicted from X(t+h)

        impute_method : str, None, or sktime transformation, optional
            Imputation method to use for missing values in the lagged data

            * default="bfill"
            * if str, admissible strings are of ``Imputer.method`` parameter, see there.
              To pass further parameters, pass the ``Imputer`` transformer directly,
              as described below.
            * if sktime transformer, this transformer is applied to the lagged data.
              This needs to be a transformer that removes missing data, and can be
              an ``Imputer``.
            * if None, no imputation is done when applying ``Lag`` transformer

        pooling : str, one of ["local", "global", "panel"], optional, default="local"
            level on which data are pooled to fit the supervised regression model
            "local" = unit/instance level, one reduced model per lowest hierarchy level
            "global" = top level, one reduced model overall, on pooled data ignoring levels
            "panel" = second lowest level, one reduced model per panel level (-2)
            if there are 2 or less levels, "global" and "panel" result in the same
            if there is only 1 level (single time series), all three settings agree

        windows_identical : bool, optional, default=False
            Specifies whether all direct models use the same number of observations
            or a different number of observations.

            * `True` : Uniform window of length (total observations - maximum
              forecasting horizon). Note: Currently, there are no missing arising
              from window length due to backwards imputation in
              `ReductionTransformer`. Without imputation, the window size
              corresponds to (total observations + 1 - window_length + maximum
              forecasting horizon).
            * `False` : Window size differs for each forecasting horizon. Window
              length corresponds to (total observations + 1 - window_length +
              forecasting horizon).
        """

    RecursiveReductionForecaster
        """Recursive reduction forecaster, incl exogeneous Rec.

        Implements recursive reduction, of forecasting to tabular regression.

        Algorithm details:

        In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
            fits ``estimator`` to feature-label pairs as defined as follows.

            features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
            ``X(t+1)``
            labels = ``y(t+1)``
            ranging over all ``t`` where the above have been observed (are in the index)

        In ``predict``, given possibly exogeneous ``X``, at cutoff time ``c``,
            applies fitted estimators' predict to
            feature = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
            ``X(c+1)``
            to obtain a prediction for ``y(c+1)``.
            If a given ``y(t)`` has not been observed, it is replaced by a prediction
            obtained in the same way - done repeatedly until all predictions are obtained.
            Out-of-sample, this results in the "recursive" behaviour, where predictions
            at time points c+1, c+2, etc, are obtained iteratively.
            In-sample, predictions are obtained in a single step, with potential
            missing values obtained via the ``impute`` strategy chosen.

        Parameters
        ----------
        estimator : sklearn regressor, must be compatible with sklearn interface
            tabular regression algorithm used in reduction algorithm

        window_length : int, optional, default=10
            window length used in the reduction algorithm

        impute_method : str, None, or sktime transformation, optional
            Imputation method to use for missing values in the lagged data

            * default="bfill"
            * if str, admissible strings are of ``Imputer.method`` parameter, see there.
              To pass further parameters, pass the ``Imputer`` transformer directly,
              as described below.
            * if sktime transformer, this transformer is applied to the lagged data.
              This needs to be a transformer that removes missing data, and can be
              an ``Imputer``.
            * if None, no imputation is done when applying ``Lag`` transformer

        pooling : str, one of ["local", "global", "panel"], optional, default="local"
            level on which data are pooled to fit the supervised regression model
            "local" = unit/instance level, one reduced model per lowest hierarchy level
            "global" = top level, one reduced model overall, on pooled data ignoring levels
            "panel" = second lowest level, one reduced model per panel level (-2)
            if there are 2 or less levels, "global" and "panel" result in the same
            if there is only 1 level (single time series), all three settings agree
        """

-----------------------------------------------------------------------------

    > "requires-fh-in-fit": True

    BaseForecaster ???
        this means it is the DEFAULT value for the models don't initialize it!!!!

    AutoTS
    EnbPIForecaster
    GreykiteForecaster
    MomentFMForecaster
    MAPAForecaster
    PyKANForecaster
    SquaringResiduals
    TimeLLMForecaster
    TinyTimeMixerForecaster

    _NeuralForecastAdapter
    _PytorchForecastingAdapter

    _DirectReducer
    _MultioutputReducer
    _DirRecReducer

    FhPlexForecaster
    GroupbyCategoryForecaster
    DirectReductionForecaster
    StackingForecaster
    TransformSelectForecaster ???
