from typing import Union, List, Optional, Dict
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class MSTL(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: Union[int, List[int]],
        trend_forecaster: dict, # =AutoETS(model="ZZN"),
        stl_kwargs: Optional[Dict] = None,
        alias: str = "MSTL",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
    ):
        super().__init__(stfm.MSTL, locals())
        return