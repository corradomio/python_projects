from typing import Optional, Tuple, Union, Literal, List

import darts.models.forecasting.arima
from darts.models.forecasting.arima import IntOrIntSequence
from darts.timeseries import TimeSeries
from sktime.forecasting.base import ForecastingHorizon

from .base import DartsBaseForecaster


class ARIMA(DartsBaseForecaster):

    def __init__(self, p: IntOrIntSequence = 12,
        d: int = 1,
        q: IntOrIntSequence = 0,
        seasonal_order: Tuple[int, IntOrIntSequence, IntOrIntSequence, int] = (
            0,
            0,
            0,
            0,
        ),
        trend: Optional[Union[Literal["n", "c", "t", "ct"], List[int]]] = None,
        random_state: Optional[int] = None,
        add_encoders: Optional[dict] = None,
    ):
        super().__init__(
            darts.models.forecasting.arima.ARIMA,
            [],
            dict(
                p=p,
                d=d,
                q=q,
                seasonal_order=seasonal_order,
                trend=trend,
                random_state=random_state,
                add_encoders=add_encoders
            )
        )
        # self.p = p
        # self.d = d
        # self.q = q
        # self.seasonal_order = seasonal_order
        # self.trend = trend
        # self.random_state = random_state
        # self.add_encoders = add_encoders
        pass
