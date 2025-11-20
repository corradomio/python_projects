Darts
-----------------------------------------------------------------------------
    Regression Models
    -----------------

    Baseline Models (LocalForecastingModel)

            NaiveMean
            NaiveSeasonal
            NaiveDrift
            NaiveMovingAverage

    Global Baseline Models (GlobalForecastingModel)

            GlobalNaiveAggregate
            GlobalNaiveDrift
            GlobalNaiveSeasonal

    Statistical Models (LocalForecastingModel)

            ARIMA
            VARIMA
            AutoARIMA
            ExponentialSmoothing
            StatsForecastModel
            AutoETS
            AutoCES
            AutoMFLES
            TBATS
            AutoTBATS
            Theta
            FourTheta
            AutoTheta
            Prophet
            FFT
            KalmanForecaster
            Croston

    SKLearn-Like Models (GlobalForecastingModel)

            SKLearnModel
            LinearRegressionModel
            RandomForestModel
            CatBoostModel
            LightGBMModel
            XGBModel

    PyTorch (Lightning)-based Models (GlobalForecastingModel)

            RNNModel
            BlockRNNModel
            NBEATSModel
            NHiTSModel
            TCNModel
            TransformerModel
            TFTModel
            DLinearModel
            NLinearModel
            TiDEModel
            TSMixerModel

    Ensemble Models (GlobalForecastingModel)

            NaiveEnsembleModel
            RegressionEnsembleModel

    Conformal Models (GlobalForecastingModel)

            ConformalNaiveModel
            ConformalQRModel

    Classification Models
    ---------------------

    SKLearn-Like Models (GlobalForecastingModel)

            SKLearnClassifierModel
            CatBoostClassifierModel
            LightGBMClassifierModel
            XGBClassifierModel

    -----------------------------------------------------------------------------

    Forecasting Models

        ARIMA
        Baseline Models
        Block Recurrent Neural Networks
        CatBoost Models
        Conformal Models
        D-Linear
        Exponential Smoothing
        Fast Fourier Transform
        Global Baseline Models (Naive)
        Kalman Filter Forecaster
        LightGBM Models
        Linear Regression Model
        N-BEATS
        N-HiTS
        N-Linear
        Facebook Prophet
        Random Forest
        Regression Ensemble Model
        Recurrent Neural Networks
        AutoARIMA
        AutoCES
        AutoETS
        AutoMFLES
        AutoTBATS
        AutoTheta
        Croston Method
        StatsForecastModel
        TBATS
        SKLearn-Like Models
        Temporal Convolutional Network
        Temporal Fusion Transformer (TFT)
        Theta Method
        Time-series Dense Encoder (TiDE)
        Transformer Model
        Time-Series Mixer (TSMixer)
        VARIMA
        XGBoost Models


Nixtla
-----------------------------------------------------------------------------

    StatsForecast               AutoARIMA, AutoETS, AutoCES, MSTL, Theta
    NeuralForecast              MLP, RNNs, NBEATS, NHITS, TFT, ...

    TimeGPT                     NO: online
    MLForecast                  NO
    HierarchicalForecast        NO


Pytorch Forecasting
-----------------------------------------------------------------------------

    All models are already available in Darts or NeuralForecast

    Models:
        RecurrentNetwork
        DecoderMLP
        NBeats
        NHiTS
        DeepAR
        TemporalFusionTransformer
        TiDEModel
