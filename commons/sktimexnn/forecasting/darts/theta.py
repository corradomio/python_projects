from typing import Optional

import darts.models.forecasting.theta as dm

from .base import _BaseDartsForecaster, SEASONALITY_MODE


class _Theta(dm.Theta):
    def __init__(
            self,
            theta: int = 2,
            seasonality_period: Optional[int] = None,
            season_mode="multiplicative",
    ):
        super().__init__(
            theta=theta,
            seasonality_period=seasonality_period,
            season_mode=SEASONALITY_MODE[season_mode]
        )

    def predict(
            self,
            n: int,
            series=None,
            num_samples: int = 1,
            verbose: Optional[bool] = None,
            show_warnings: bool = True,
            random_state: Optional[int] = None,
    ) -> "TimeSeries":
        return super().predict(
            n=n,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            random_state=random_state
        )


class Theta(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

    def __init__(
            self,
            theta: int = 2,
            seasonality_period: Optional[int] = None,
            season_mode: str = "multiplicative",
            # --
            scaler=None,
            # --
    ):
        super().__init__(_Theta, locals())
