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
        verbose: bool = False,
    ):
        super().__init__(stfm.AutoARIMA, locals())
        return


class ARIMA(_BaseStatsForecastForecaster):
    def __init__(
        self,
        # p,d,q
        # order: Tuple[int, int, int] = (0, 0, 0),
        order: Tuple[int, int, int]|None = None,
        p: int = 0,
        d: int = 0,
        q: int = 0,

        # P,D,Q
        # seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int]|None = None,
        P: int = 0,
        D: int = 0,
        Q: int = 0,

        # S
        season_length: int = 1,

        include_mean: bool = True,
        include_drift: bool = False,
        include_constant: Optional[bool] = None,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        method: str = "CSS-ML",
        fixed: Optional[dict] = None,

        # alias: str = "ARIMA",
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(stfm.ARIMA, locals())
        return

    def _validate_kwargs(self, stf_kwargs: dict, y, X) -> dict:
        stf_kwargs = {} | stf_kwargs
        if stf_kwargs["order"] is None:
            p = stf_kwargs["p"]
            d = stf_kwargs["d"]
            q = stf_kwargs["q"]
            stf_kwargs["order"] = [p,d,q]

        del stf_kwargs["p"]
        del stf_kwargs["d"]
        del stf_kwargs["q"]

        if stf_kwargs["seasonal_order"] is None:
            P = stf_kwargs["P"]
            D = stf_kwargs["D"]
            Q = stf_kwargs["Q"]
            stf_kwargs["seasonal_order"] = [P,D,Q]

        del stf_kwargs["P"]
        del stf_kwargs["D"]
        del stf_kwargs["Q"]

        return stf_kwargs
