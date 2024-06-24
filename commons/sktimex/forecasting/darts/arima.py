
__all__ = [
    "DartsARIMAForecaster"
]

from typing import Optional, Tuple, Union, Literal, List
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.arima import IntOrIntSequence

from .base import BaseDartsForecaster


# ---------------------------------------------------------------------------

class DartsARIMAForecaster(BaseDartsForecaster):

    def __init__(
        self, *,
        p: IntOrIntSequence = 12,
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
        kwargs = dict(
            p=p,
            d=d,
            q=q,
            seasonal_order=seasonal_order,
            trend=trend,
            random_state=random_state,
            add_encoders=add_encoders
        )
        super().__init__(
            ARIMA,
            kwargs
        )
        pass

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y, None, None)

    def _predict(self, fh, X=None):
        return super()._predict(fh, None)
