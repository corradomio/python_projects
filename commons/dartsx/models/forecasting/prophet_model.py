from typing import Optional, Union, Callable, Sequence, Any
import pandas as pd
import darts.models.forecasting.prophet_model as dm

from .base import _BaseDartsForecaster


class Prophet(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

    def __init__(
            self,
            add_seasonalities: Optional[Union[dict, list[dict]]] = None,
            add_regressor_configs: Optional[dict[str, dict[str, Any]]] = None,
            country_holidays: Optional[str] = None,
            cap: Optional[
                Union[
                    float,
                    Callable[[Union[pd.DatetimeIndex, pd.RangeIndex]], Sequence[float]],
                ]
            ] = None,
            floor: Optional[
                Union[
                    float,
                    Callable[[Union[pd.DatetimeIndex, pd.RangeIndex]], Sequence[float]],
                ]
            ] = None,
            add_encoders: Optional[dict] = None,
            random_state: Optional[int] = None,
            suppress_stdout_stderror: bool = True,
            # --
            scaler=None,
            # --
            **kwargs,
    ):
        super().__init__(dm.Prophet, locals())
