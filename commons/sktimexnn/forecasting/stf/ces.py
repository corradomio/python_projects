from typing import Optional, Dict
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoCES(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: int = 1,
        model: str = "Z",
        alias: str = "CES",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(stfm.AutoCES, locals())
        return
