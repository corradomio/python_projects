from typing import Optional, Dict
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoTheta(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = None,
        alias: str = "AutoTheta",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
    ):
        super().__init__(stfm.AutoTheta, locals())
        return

