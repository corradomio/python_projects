from typing import Optional, Union, List, Dict, Any
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoMFLES(_BaseStatsForecastForecaster):
    def __init__(
        self,
        test_size: int,
        season_length: Optional[Union[int, List[int]]] = None,
        n_windows: int = 2,
        config: Optional[Dict[str, Any]] = None,
        step_size: Optional[int] = None,
        metric: str = "smape",
        verbose: bool = False,
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        alias: str = "AutoMFLES",
    ):
        super().__init__(stfm.AutoMFLES, locals())
        return


class MFLES(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: Optional[Union[int, List[int]]] = None,
        fourier_order: Optional[int] = None,
        max_rounds: int = 50,
        ma: Optional[int] = None,
        alpha: float = 1.0,
        decay: float = -1.0,
        changepoints: bool = True,
        n_changepoints: Union[float, int] = 0.25,
        seasonal_lr: float = 0.9,
        trend_lr: float = 0.9,
        exogenous_lr: float = 1.0,
        residuals_lr: float = 1.0,
        cov_threshold: float = 0.7,
        moving_medians: bool = False,
        min_alpha: float = 0.05,
        max_alpha: float = 1.0,
        trend_penalty: bool = True,
        multiplicative: Optional[bool] = None,
        smoother: bool = False,
        robust: Optional[bool] = None,
        verbose: bool = False,
        # prediction_intervals: Optional[ConformalIntervals] = None,
        prediction_intervals: Optional[Dict] = None,
        alias: str = "MFLES",
    ):
        super().__init__(stfm.MFLES, locals())
        return