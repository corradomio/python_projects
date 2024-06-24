sktime interface
----------------

    _is_fitted
    _y
    _X
    _fh
    _cutoff

    -------------------------------------------------------------------------

    cutoff
    fh
    is_fitted

    -------------------------------------------------------------------------

    fit(y[, X, fh])
    fit_predict(y[, X, fh, X_pred])

    predict(fh [, X]) | predict([fh, X])

    predict_interval([fh, X, coverage])
    predict_proba([fh, X, marginal])
    predict_quantiles([fh, X, alpha])
    predict_residuals([y, X])
    predict_var([fh, X, cov])

    score(y[, X, fh])

    update(y[, X, update_params])
    update_predict(y[, cv, X, update_params, ...])
    update_predict_single([y, fh, X, update_params])

    -------------------------------------------------------------------------
    Note: it is added automatically this method

    predict_history(fh [,X], y_past [,X_past][, update_params])
        update(yhisto[, Xhisto, update_params])
        predict(fh [, X])


Come estendere sktime
---------------------

    una classe sktime deve essere in grado di clonare se stessa in diverse situazioni.
    Per fare cio' servono 2 cose: i nome della classe, ma questo non e' un problema
    i parametri, che vengono ricuperati con 
    
        
        _FlagManager (skbase.base._tagmanager)      
            """Mixin class for flag and configuration settings management."""
            
            BaseObject(_FlagManager) (skbase.base._base)
                """Base class for parametric objects with sktime style tag interface.
                Extends scikit-learn's BaseEstimator to include sktime style interface for tags.
                """
                
                BaseObject(_BaseObject) (sktime.base._base)
                    """Base class for parametric objects with tags in sktime.
                    Extends skbase BaseObject with additional features.
                    """
                    
                    BaseEstimator(BaseObject) (sktime.base._base)
                        """Base class for defining estimators in sktime.
                        Extends sktime's BaseObject to include basic functionality for fittable estimators.
                        """
                        
                        BaseForecaster(BaseEstimator) (sktime.forecasting.base._base)
                            """Base forecaster template class.
                            The base forecaster specifies the methods and method signatures that all forecasters
                            have to implement.
                            Specific implementations of these methods is deferred to concrete forecasters.
                            """

sktime.datatypes.MTYPE_REGISTER
-------------------------------

    ('pd.Series',           'Series', 'pd.Series representation of a univariate series')
    ('pd.DataFrame',        'Series', 'pd.DataFrame representation of a uni- or multivariate series')
    ('np.ndarray',          'Series', '2D numpy.ndarray with rows=samples, cols=variables, index=integers')
    ('xr.DataArray',        'Series', 'xr.DataArray representation of a uni- or multivariate series')
    ('dask_series',         'Series', 'xdas representation of a uni- or multivariate series')
    ('nested_univ',         'Panel', 'pd.DataFrame with one column per variable, pd.Series in cells')
    ('numpy3D', 'Panel',    '3D np.array of format (n_instances, n_columns, n_timepoints)')
    ('pd-multiindex',       'Panel', 'pd.DataFrame with multi-index (instances, timepoints)')
    ('pd-wide', 'Panel',    'pd.DataFrame in wide format, cols = (instance*timepoints)')
    ('pd-long', 'Panel',    'pd.DataFrame in long format, cols = (index, time_index, column)')
    ('df-list', 'Panel',    'list of pd.DataFrame')
    ('dask_panel',          'Panel', 'dask frame with one instance and one time index, as per dask_to_pd convention')
    ('pd_multiindex_hier',  'Hierarchical', 'pd.DataFrame with MultiIndex')
    ('dask_hierarchical',   'Hierarchical', 'dask frame with multiple hierarchical indices, as per dask_to_pd convention')
    ('alignment',           'Alignment', 'pd.DataFrame in alignment format, values are iloc index references')
    ('alignment_loc',       'Alignment', 'pd.DataFrame in alignment format, values are loc index references')
    
    ('pd_DataFrame_Table',  'Table', 'pd.DataFrame representation of a data table')
    ('numpy1D',             'Table', '1D np.narray representation of a univariate data table')
    ('numpy2D',             'Table', '2D np.narray representation of a multivariate data table')
    ('pd_Series_Table',     'Table', 'pd.Series representation of a univariate data table')
    ('list_of_dict',        'Table', 'list of dictionaries with primitive entries')
    ('polars_eager_table',  'Table', 'polars.DataFrame representation of a data table')
    ('polars_lazy_table',   'Table', 'polars.LazyFrame representation of a data table')
    ('pred_interval',       'Proba', 'predictive intervals')
    ('pred_quantiles',      'Proba', 'quantile predictions')
    ('pred_var',            'Proba', 'variance predictions')
    
    ('numpyflat', 'Panel', 'WARNING: only for internal use, not a fully supported Panel mtype. 2D np.array of format (n_instances, n_columns*n_timepoints)')




sktime Hierarchy
----------------

BaseEstimator(BaseObject) (sktime.base._base)
    BaseForecaster(BaseEstimator) (sktime.forecasting.base._base)

        MyForecaster(BaseForecaster) (extension_templates.forecasting_simple)
        MyForecaster(BaseForecaster) (extension_templates.forecasting)
        MyForecaster(BaseForecaster) (extension_templates.forecasting_supersimple)

        MockUnivariateForecasterLogger(BaseForecaster, _MockEstimatorMixin) (sktime.utils.estimators._forecasters)
        MockForecaster(BaseForecaster) (sktime.utils.estimators._forecasters)
        BaseDeepNetworkPyTorch(BaseForecaster, ABC) (sktime.networks.base)
        ReconcilerForecaster(BaseForecaster) (sktime.forecasting.reconcile)
        Croston(BaseForecaster) (sktime.forecasting.croston)
        SquaringResiduals(BaseForecaster) (sktime.forecasting.squaring_residuals)
        ForecastKnownValues(BaseForecaster) (sktime.forecasting.dummy)
        NaiveVariance(BaseForecaster) (sktime.forecasting.naive)
        ConformalIntervals(BaseForecaster) (sktime.forecasting.conformal)
        HFTransformersForecaster(BaseForecaster) (sktime.forecasting.hf_transformers_forecaster)
        ThetaModularForecaster(BaseForecaster) (sktime.forecasting.theta)
        ARCH(BaseForecaster) (sktime.forecasting.arch._uarch)
        _HeterogenousEnsembleForecaster(_HeterogenousMetaEstimator, BaseForecaster) (sktime.forecasting.base._meta)
            AutoEnsembleForecaster(_HeterogenousEnsembleForecaster) (sktime.forecasting.compose._ensemble)
            EnsembleForecaster(_HeterogenousEnsembleForecaster) (sktime.forecasting.compose._ensemble)
                OnlineEnsembleForecaster(EnsembleForecaster) (sktime.forecasting.online_learning._online_ensemble)
            ColumnEnsembleForecaster(_HeterogenousEnsembleForecaster, _ColumnEstimator) (sktime.forecasting.compose._column_ensemble)
            StackingForecaster(_HeterogenousEnsembleForecaster) (sktime.forecasting.compose._stack)
            HierarchyEnsembleForecaster(_HeterogenousEnsembleForecaster) (sktime.forecasting.compose._hierarchy_ensemble)
        _DelegatedForecaster(BaseForecaster) (sktime.forecasting.base._delegate)
            PluginParamsForecaster(_DelegatedForecaster) (sktime.param_est.plugin._forecaster)
            UpdateRefitsEvery(_DelegatedForecaster) (sktime.forecasting.stream._update)
            UpdateEvery(_DelegatedForecaster) (sktime.forecasting.stream._update)
            DontUpdate(_DelegatedForecaster) (sktime.forecasting.stream._update)
            IgnoreX(_DelegatedForecaster) (sktime.forecasting.compose._ignore_x)
            MultiplexForecaster(_HeterogenousMetaEstimator, _DelegatedForecaster) (sktime.forecasting.compose._multiplexer)
            Permute(_DelegatedForecaster, BaseForecaster, _HeterogenousMetaEstimator) (sktime.forecasting.compose._pipeline)
            ForecastByLevel(_DelegatedForecaster) (sktime.forecasting.compose._grouped)
            FallbackForecaster(_HeterogenousMetaEstimator, _DelegatedForecaster) (sktime.forecasting.compose._fallback)
            BaseGridSearch(_DelegatedForecaster) (sktime.forecasting.model_selection._tune)
                ForecastingGridSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._tune)
                ForecastingRandomizedSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._tune)
                ForecastingSkoptSearchCV(BaseGridSearch) (sktime.forecasting.model_selection._tune)
        _BaseWindowForecaster(BaseForecaster) (sktime.forecasting.base._sktime)
            NaiveForecaster(_BaseWindowForecaster) (sktime.forecasting.naive)
            _Reducer(_BaseWindowForecaster) (sktime.forecasting.compose._reduce)
                _DirectReducer(_Reducer) (sktime.forecasting.compose._reduce)
                    DirectTabularRegressionForecaster(_DirectReducer) (sktime.forecasting.compose._reduce)
                    DirectTimeSeriesRegressionForecaster(_DirectReducer) (sktime.forecasting.compose._reduce)
                _MultioutputReducer(_Reducer) (sktime.forecasting.compose._reduce)
                    MultioutputTabularRegressionForecaster(_MultioutputReducer) (sktime.forecasting.compose._reduce)
                    MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer) (sktime.forecasting.compose._reduce)
                _RecursiveReducer(_Reducer) (sktime.forecasting.compose._reduce)
                    RecursiveTabularRegressionForecaster(_RecursiveReducer) (sktime.forecasting.compose._reduce)
                    RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer) (sktime.forecasting.compose._reduce)
                _DirRecReducer(_Reducer) (sktime.forecasting.compose._reduce)
                    DirRecTabularRegressionForecaster(_DirRecReducer) (sktime.forecasting.compose._reduce)
                    DirRecTimeSeriesRegressionForecaster(_DirRecReducer) (sktime.forecasting.compose._reduce)
        BaseDeepNetworkPyTorch(BaseForecaster, ABC) (sktime.forecasting.base.adapters._pytorch)
            LTSFLinearForecaster(BaseDeepNetworkPyTorch) (sktime.forecasting.ltsf)
            LTSFDLinearForecaster(BaseDeepNetworkPyTorch) (sktime.forecasting.ltsf)
            LTSFNLinearForecaster(BaseDeepNetworkPyTorch) (sktime.forecasting.ltsf)
            CINNForecaster(BaseDeepNetworkPyTorch) (sktime.forecasting.conditional_invertible_neural_network)
        _StatsForecastAdapter(BaseForecaster) (sktime.forecasting.base.adapters._statsforecast)
        _GeneralisedStatsForecastAdapter(BaseForecaster) (sktime.forecasting.base.adapters._generalised_statsforecast)
            StatsForecastAutoARIMA(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastAutoTheta(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastAutoETS(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastAutoCES(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastAutoTBATS(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastMSTL(_GeneralisedStatsForecastAdapter) (sktime.forecasting.statsforecast)
            StatsForecastGARCH(_GeneralisedStatsForecastAdapter) (sktime.forecasting.arch._statsforecast_arch)
            StatsForecastARCH(_GeneralisedStatsForecastAdapter) (sktime.forecasting.arch._statsforecast_arch)
        _NeuralForecastAdapter(BaseForecaster) (sktime.forecasting.base.adapters._neuralforecast)
            NeuralForecastRNN(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
            NeuralForecastLSTM(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
        _ProphetAdapter(BaseForecaster) (sktime.forecasting.base.adapters._fbprophet)
            Prophet(_ProphetAdapter) (sktime.forecasting.fbprophet)
            ProphetPiecewiseLinearTrendForecaster(_ProphetAdapter) (sktime.forecasting.trend._pwl_trend_forecaster)
        _PmdArimaAdapter(BaseForecaster) (sktime.forecasting.base.adapters._pmdarima)
            AutoARIMA(_PmdArimaAdapter) (sktime.forecasting.arima._pmdarima)
            ARIMA(_PmdArimaAdapter) (sktime.forecasting.arima._pmdarima)
        _TbatsAdapter(BaseForecaster) (sktime.forecasting.base.adapters._tbats)
            TBATS(_TbatsAdapter) (sktime.forecasting.tbats)
            BATS(_TbatsAdapter) (sktime.forecasting.bats)
        _StatsModelsAdapter(BaseForecaster) (sktime.forecasting.base.adapters._statsmodels)
            AutoETS(_StatsModelsAdapter) (sktime.forecasting.ets)
            VECM(_StatsModelsAdapter) (sktime.forecasting.vecm)
            DynamicFactor(_StatsModelsAdapter) (sktime.forecasting.dynamic_factor)
            VAR(_StatsModelsAdapter) (sktime.forecasting.var)
            ExponentialSmoothing(_StatsModelsAdapter) (sktime.forecasting.exp_smoothing)
                ThetaForecaster(ExponentialSmoothing) (sktime.forecasting.theta)
            ARDL(_StatsModelsAdapter) (sktime.forecasting.ardl)
            SARIMAX(_StatsModelsAdapter) (sktime.forecasting.sarimax)
            VARMAX(_StatsModelsAdapter) (sktime.forecasting.varmax)
            AutoREG(_StatsModelsAdapter) (sktime.forecasting.auto_reg)
            UnobservedComponents(_StatsModelsAdapter) (sktime.forecasting.structural)
            StatsModelsARIMA(_StatsModelsAdapter) (sktime.forecasting.arima._statsmodels)
        STLForecaster(BaseForecaster) (sktime.forecasting.trend._stl_forecaster)
        TrendForecaster(BaseForecaster) (sktime.forecasting.trend._trend_forecaster)
        CurveFitForecaster(BaseForecaster) (sktime.forecasting.trend._curve_fit_forecaster)
        PolynomialTrendForecaster(BaseForecaster) (sktime.forecasting.trend._polynomial_trend_forecaster)
        BaggingForecaster(BaseForecaster) (sktime.forecasting.compose._bagging)
        DirectReductionForecaster(BaseForecaster, _ReducerMixin) (sktime.forecasting.compose._reduce)
        RecursiveReductionForecaster(BaseForecaster, _ReducerMixin) (sktime.forecasting.compose._reduce)
        YfromX(BaseForecaster, _ReducerMixin) (sktime.forecasting.compose._reduce)
        _Pipeline(_HeterogenousMetaEstimator, BaseForecaster) (sktime.forecasting.compose._pipeline)
            ForecastingPipeline(_Pipeline) (sktime.forecasting.compose._pipeline)
            TransformedTargetForecaster(_Pipeline) (sktime.forecasting.compose._pipeline)
        ForecastX(BaseForecaster) (sktime.forecasting.compose._pipeline)
        Permute(_DelegatedForecaster, BaseForecaster, _HeterogenousMetaEstimator) (sktime.forecasting.compose._pipeline)
        FhPlexForecaster(BaseForecaster) (sktime.forecasting.compose._fhplex)
        DummyForecaster(_HeterogenousMetaEstimator, BaseForecaster) (sktime.forecasting.compose.tests.test_fallback)
        HCrystalBallAdapter(BaseForecaster) (sktime.forecasting.adapters._hcrystalball)

