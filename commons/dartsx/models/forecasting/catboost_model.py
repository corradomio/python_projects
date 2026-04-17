from typing import Optional, Union

import darts.models.forecasting.catboost_model as dm

from .base import _BaseDartsForecaster


class CatBoostModel(_BaseDartsForecaster):

    def __init__(
            self,
            lags: Union[int, list] = None,
            lags_past_covariates: Union[int, list[int]] = None,
            lags_future_covariates: Union[tuple[int, int], list[int]] = None,
            output_chunk_length: int = 1,
            output_chunk_shift: int = 0,
            add_encoders: Optional[dict] = None,
            likelihood: Optional[str] = None,
            quantiles: list = None,
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
        super().__init__(dm.CatBoostModel, locals())
