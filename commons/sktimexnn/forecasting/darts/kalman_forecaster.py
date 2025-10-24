from typing import Optional

import darts.models.forecasting.kalman_forecaster as dm
from nfoursid.kalman import Kalman

from .base import _BaseDartsForecaster


class KalmanForecaster(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

    def __init__(
            self,
            dim_x: int = 1,
            kf: Optional[Kalman] = None,
            add_encoders: Optional[dict] = None,
            random_state: Optional[int] = None,
            # --
            scaler=None,
            # --
    ):
        super().__init__(dm.KalmanForecaster, locals())
