It is Darts that uses TimeSeries!


polars

it theory it supports datetime in index (with warning)


NeuralForecast
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
