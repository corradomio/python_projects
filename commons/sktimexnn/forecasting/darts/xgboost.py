from typing import Tuple, Optional, Union, Literal, List, TypeAlias, Sequence

import darts.models.forecasting.xgboost as dm
from darts.models.forecasting.sklearn_model import LAGS_TYPE, FUTURE_LAGS_TYPE

from .base import _BaseDartsForecaster


IntOrIntSequence: TypeAlias = Union[int, Sequence[int]]


class XGBModel(_BaseDartsForecaster):

    # _tags = {
    #     "capability:exogenous": False,
    #     "capability:future-exogenous": False
    # }

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
            # --
            scaler=None,
            # --
            **kwargs,
        ):
        super().__init__(dm.XGBModel, locals())
        return
