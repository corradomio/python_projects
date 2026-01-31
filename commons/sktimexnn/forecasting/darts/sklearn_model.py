from typing import Optional
from stdlib.qname import create_from, create_from_collection

import darts.models.forecasting.sklearn_model as dm
from darts.models.forecasting.sklearn_model import LAGS_TYPE, FUTURE_LAGS_TYPE

from .base import _BaseDartsForecaster


class _SKLearnModel(dm.SKLearnModel):
    def __init__(
            self,
            lags: Optional[LAGS_TYPE] = None,
            lags_past_covariates: Optional[LAGS_TYPE] = None,
            lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
            output_chunk_length: int = 1,
            output_chunk_shift: int = 0,
            add_encoders: Optional[dict] = None,
            model: dict=None,
            multi_models: Optional[bool] = True,
            use_static_covariates: bool = True,
            random_state: Optional[int] = None,
    ):
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=create_from_collection(add_encoders),
            model=create_from(model),
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state
        )


class SKLearnModel(_BaseDartsForecaster):
    def __init__(
            self,
            lags: Optional[LAGS_TYPE] = None,
            lags_past_covariates: Optional[LAGS_TYPE] = None,
            lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
            output_chunk_length: int = 1,
            output_chunk_shift: int = 0,
            add_encoders: Optional[dict] = None,
            model=None,
            multi_models: Optional[bool] = True,
            use_static_covariates: bool = True,
            random_state: Optional[int] = None,
            # --
            scaler=None,
            # --
    ):
        super().__init__(_SKLearnModel, locals())
