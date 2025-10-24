from typing import Optional

import darts.models.forecasting.random_forest as dm
from darts.models.forecasting.sklearn_model import LAGS_TYPE, FUTURE_LAGS_TYPE

from .base import _BaseDartsForecaster


class RandomForestModel(_BaseDartsForecaster):
    def __init__(
            self,
            lags: Optional[LAGS_TYPE] = None,
            lags_past_covariates: Optional[LAGS_TYPE] = None,
            lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
            output_chunk_length: int = 1,
            output_chunk_shift: int = 0,
            add_encoders: Optional[dict] = None,
            n_estimators: Optional[int] = 100,
            max_depth: Optional[int] = None,
            multi_models: Optional[bool] = True,
            use_static_covariates: bool = True,
            random_state: Optional[int] = None,
            # --
            scaler=None,
            # --
            **kwargs,
    ):
        super().__init__(dm.RandomForestModel, locals())
