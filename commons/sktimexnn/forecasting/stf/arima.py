from typing import Optional, Dict, Tuple
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoARIMA(_BaseStatsForecastForecaster):
    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        stepwise: bool = True,
        nmodels: int = 94,
        trace: bool = False,
        approximation: Optional[bool] = False,
        method: Optional[str] = None,
        truncate: Optional[bool] = None,
        test: str = "kpss",
        test_kwargs: Optional[str] = None,
        seasonal_test: str = "seas",
        seasonal_test_kwargs: Optional[Dict] = None,
        allowdrift: bool = True,
        allowmean: bool = True,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        season_length: int = 1,
        alias: str = "AutoARIMA",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
    ):
        super().__init__(stfm.AutoARIMA, locals())
        return


class ARIMA(_BaseStatsForecastForecaster):
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        season_length: int = 1,
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        include_mean: bool = True,
        include_drift: bool = False,
        include_constant: Optional[bool] = None,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        method: str = "CSS-ML",
        fixed: Optional[dict] = None,
        alias: str = "ARIMA",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
    ):
        super().__init__(stfm.ARIMA, locals())
        return
