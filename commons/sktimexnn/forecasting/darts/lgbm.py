from typing import Optional, Union

import darts.models.forecasting.lgbm as dm
from darts.models.forecasting.sklearn_model import LAGS_TYPE, FUTURE_LAGS_TYPE

from .base import _BaseDartsForecaster


class LightGBMModel(_BaseDartsForecaster):
    def __init__(
            self,
            lags: Optional[LAGS_TYPE] = None,
            lags_past_covariates: Optional[LAGS_TYPE] = None,
            lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
            output_chunk_length: int = 1,
            output_chunk_shift: int = 0,
            add_encoders: Optional[dict] = None,
            likelihood: Optional[str] = None,
            quantiles: Optional[list[float]] = None,
            random_state: Optional[int] = None,
            multi_models: Optional[bool] = True,
            use_static_covariates: bool = True,
            categorical_past_covariates: Optional[Union[str, list[str]]] = None,
            categorical_future_covariates: Optional[Union[str, list[str]]] = None,
            categorical_static_covariates: Optional[Union[str, list[str]]] = None,
            # --
            scaler=None,
            # --
            **kwargs,
    ):
        super().__init__(dm.LightGBMModel, locals())
