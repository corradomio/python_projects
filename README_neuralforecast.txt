Notes
-----

    It is Darts that uses TimeSeries!


    polars

    it theory it supports datetime in index (with warning)


NeuralForecast
--------------
    def __init__(
        self,
        models: List[Any],
        freq: Union[str, int],
        local_scaler_type: Optional[str] = None,
    ):

    def fit(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        val_size: Optional[int] = 0,
        sort_df: bool = True,
        use_init_models: bool = False,
        verbose: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        distributed_config: Optional[DistributedConfig] = None,
    ) -> None:

    def predict(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        sort_df: bool = True,
        verbose: bool = False,
        engine=None,
        **data_kwargs,
    ):

    def predict(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        sort_df: bool = True,
        verbose: bool = False,
        engine=None,
        **data_kwargs,
    ):



Hierarchy
---------

BaseModel(pl.LightningModule) (neuralforecast.common._base_model)
    Autoformer(BaseModel) (neuralforecast.models.autoformer)
    BiTCN(BaseModel) (neuralforecast.models.bitcn)
    DLinear(BaseModel) (neuralforecast.models.dlinear)
    DeepAR(BaseModel) (neuralforecast.models.deepar)
    DeepNPTS(BaseModel) (neuralforecast.models.deepnpts)
    DilatedRNN(BaseModel) (neuralforecast.models.dilated_rnn)
    FEDformer(BaseModel) (neuralforecast.models.fedformer)
    GRU(BaseModel) (neuralforecast.models.gru)
    Informer(BaseModel) (neuralforecast.models.informer)
    KAN(BaseModel) (neuralforecast.models.kan)
    LSTM(BaseModel) (neuralforecast.models.lstm)
    MLP(BaseModel) (neuralforecast.models.mlp)
    MLPMultivariate(BaseModel) (neuralforecast.models.mlpmultivariate)
    NBEATS(BaseModel) (neuralforecast.models.nbeats)
    NBEATSx(BaseModel) (neuralforecast.models.nbeatsx)
    NHITS(BaseModel) (neuralforecast.models.nhits)
    NLinear(BaseModel) (neuralforecast.models.nlinear)
    PatchTST(BaseModel) (neuralforecast.models.patchtst)
    RMoK(BaseModel) (neuralforecast.models.rmok)
    RNN(BaseModel) (neuralforecast.models.rnn)
    SOFTS(BaseModel) (neuralforecast.models.softs)
    StemGNN(BaseModel) (neuralforecast.models.stemgnn)
    TCN(BaseModel) (neuralforecast.models.tcn)
    TFT(BaseModel) (neuralforecast.models.tft)
    TSMixer(BaseModel) (neuralforecast.models.tsmixer)
    TSMixerx(BaseModel) (neuralforecast.models.tsmixerx)
    TiDE(BaseModel) (neuralforecast.models.tide)
    TimeLLM(BaseModel) (neuralforecast.models.timellm)
    TimeMixer(BaseModel) (neuralforecast.models.timemixer)
    TimeXer(BaseModel) (neuralforecast.models.timexer)
    TimesNet(BaseModel) (neuralforecast.models.timesnet)
    VanillaTransformer(BaseModel) (neuralforecast.models.vanillatransformer)
    iTransformer(BaseModel) (neuralforecast.models.itransformer)
    xLSTM(BaseModel) (neuralforecast.models.xlstm)

    DummyUnivariate(BaseModel) (tests.dummy.dummy_models)
    DummyMultivariate(BaseModel) (tests.dummy.dummy_models)
    DummyRecurrent(BaseModel) (tests.dummy.dummy_models)


BaseAuto(pl.LightningModule) (neuralforecast.common._base_auto)
    AutoRNN(BaseAuto) (neuralforecast.auto)
    AutoLSTM(BaseAuto) (neuralforecast.auto)
    AutoGRU(BaseAuto) (neuralforecast.auto)
    AutoTCN(BaseAuto) (neuralforecast.auto)
    AutoDeepAR(BaseAuto) (neuralforecast.auto)
    AutoDilatedRNN(BaseAuto) (neuralforecast.auto)
    AutoBiTCN(BaseAuto) (neuralforecast.auto)
    AutoxLSTM(BaseAuto) (neuralforecast.auto)
    AutoMLP(BaseAuto) (neuralforecast.auto)
    AutoNBEATS(BaseAuto) (neuralforecast.auto)
    AutoNBEATSx(BaseAuto) (neuralforecast.auto)
    AutoNHITS(BaseAuto) (neuralforecast.auto)
    AutoDLinear(BaseAuto) (neuralforecast.auto)
    AutoNLinear(BaseAuto) (neuralforecast.auto)
    AutoTiDE(BaseAuto) (neuralforecast.auto)
    AutoDeepNPTS(BaseAuto) (neuralforecast.auto)
    AutoKAN(BaseAuto) (neuralforecast.auto)
    AutoTFT(BaseAuto) (neuralforecast.auto)
    AutoVanillaTransformer(BaseAuto) (neuralforecast.auto)
    AutoInformer(BaseAuto) (neuralforecast.auto)
    AutoAutoformer(BaseAuto) (neuralforecast.auto)
    AutoFEDformer(BaseAuto) (neuralforecast.auto)
    AutoPatchTST(BaseAuto) (neuralforecast.auto)
    AutoiTransformer(BaseAuto) (neuralforecast.auto)
    AutoTimeXer(BaseAuto) (neuralforecast.auto)
    AutoTimesNet(BaseAuto) (neuralforecast.auto)
    AutoStemGNN(BaseAuto) (neuralforecast.auto)
    AutoHINT(BaseAuto) (neuralforecast.auto)
    AutoTSMixer(BaseAuto) (neuralforecast.auto)
    AutoTSMixerx(BaseAuto) (neuralforecast.auto)
    AutoMLPMultivariate(BaseAuto) (neuralforecast.auto)
    AutoSOFTS(BaseAuto) (neuralforecast.auto)
    AutoTimeMixer(BaseAuto) (neuralforecast.auto)
    AutoRMoK(BaseAuto) (neuralforecast.auto)
