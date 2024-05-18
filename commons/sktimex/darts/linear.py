from typing import Optional, Tuple, Union, Literal, List, Dict

import darts.models.forecasting.linear_regression_model
from darts.timeseries import TimeSeries
from sktime.forecasting.base import ForecastingHorizon

from .base import DartsBaseForecaster

LAGS_TYPE = Union[int, List[int], Dict[str, Union[int, List[int]]]]
FUTURE_LAGS_TYPE = Union[
    Tuple[int, int], List[int], Dict[str, Union[Tuple[int, int], List[int]]]
]


class LinearForecaster(DartsBaseForecaster):

    def __init__(self, lags: Optional[LAGS_TYPE] = None,
                 lags_past_covariates: Optional[LAGS_TYPE] = None,
                 lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
                 output_chunk_length: int = 1,
                 output_chunk_shift: int = 0,
                 add_encoders: Optional[dict] = None,
                 likelihood: Optional[str] = None,
                 quantiles: Optional[List[float]] = None,
                 random_state: Optional[int] = None,
                 multi_models: Optional[bool] = True,
                 use_static_covariates: bool = True,
                 **kwargs
    ):
        super().__init__(
            darts.models.forecasting.linear_regression_model.LinearRegressionModel,
            [],
            dict(
                lags=lags,
                lags_past_covariates=lags_past_covariates,
                lags_future_covariates=lags_future_covariates,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                add_encoders=add_encoders,
                likelihood=likelihood,
                quantiles=quantiles,
                random_state=random_state,
                multi_models=multi_models,
                use_static_covariates=use_static_covariates,
                **kwargs
            )
        )