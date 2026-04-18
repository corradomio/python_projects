from typing import Union, List, Optional
from .base import _BaseStatsForecastForecaster
import statsforecast.models as stfm


class AutoTBATS(_BaseStatsForecastForecaster):
    def __init__(
        self,
        season_length: Union[int, List[int]] = 1,
        use_boxcox: Optional[bool] = None,
        bc_lower_bound: float = 0.0,
        bc_upper_bound: float = 1.0,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        use_arma_errors: bool = True,
        # alias: str = "AutoTBATS",
        verbose: bool = False,
    ):
        super().__init__(stfm.AutoTBATS, locals())
        return
