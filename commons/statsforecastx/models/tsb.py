from typing import Optional, Dict
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class TSB(_BaseStatsForecastForecaster):
    def __init__(
        self,
        alpha_d: float,
        alpha_p: float,
        alias: str = "TSB",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(stfm.TSB, locals())
        return
