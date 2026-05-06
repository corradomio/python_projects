from typing import Tuple, Optional, Union, Literal, List, TypeAlias, Sequence

import darts.models.forecasting.arima as dm

from .base import _BaseDartsForecaster


IntOrIntSequence: TypeAlias = Union[int, Sequence[int]]

class SARIMA(dm.ARIMA):
    def __init__(
        self,
        p=12,d=1,q=0,
        P=0, D=0, Q=0, S=0,
        trend=None,
        random_state=None,
        add_encoders=None
    ):
        super().__init__(
            p=p,
            d=d,
            q=q,
            seasonal_order=[P,D,Q,S],
            trend=trend,
            random_state=random_state,
            add_encoders=add_encoders
        )


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
        P: int = 0,
        D: IntOrIntSequence = 0,
        Q: IntOrIntSequence = 0,
        S: int = 0,
        # seasonal_order: Tuple[int, IntOrIntSequence, IntOrIntSequence, int] = (
        #     0,
        #     0,
        #     0,
        #     0,
        # ),
        trend: Optional[Union[Literal["n", "c", "t", "ct"], List[int]]] = None,
        random_state: Optional[int] = None,
        add_encoders: Optional[dict] = None,
        # --
        scaler=None,
        # --
        **kwargs
    ):
        super().__init__(SARIMA, locals())
        return
