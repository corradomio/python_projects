from typing import Optional, Dict
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoETS(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        alias: str = "AutoETS",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(stfm.AutoETS, locals())
        return

