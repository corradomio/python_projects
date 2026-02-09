from typing import Tuple, Optional, Union, Literal, List, TypeAlias, Sequence

import darts.models.forecasting.arima as dm

from .base import _BaseDartsForecaster


IntOrIntSequence: TypeAlias = Union[int, Sequence[int]]


class ARIMA(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

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
            # --
            scaler=None,
            # --
            **kwargs
        ):
        super().__init__(dm.ARIMA, locals())
        return
