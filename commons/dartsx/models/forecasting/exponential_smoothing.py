from typing import Optional, Any

import darts.models.forecasting.exponential_smoothing as dm

from .base import _BaseDartsForecaster, TREND_MODE, SEASONALITY_MODE, MODEL_MODE


class _ExponentialSmoothing(dm.ExponentialSmoothing):
    # For compatibility with the general implementation,
    # the original 'kwargs' is renamed in 'init_kwargs', and
    # the original '**fit_kwargs' in '**kwargs'
    def __init__(
            self,
            trend: Optional[str] = "additive",
            damped: Optional[bool] = False,
            seasonal: Optional[str] = "additive",
            seasonal_periods: Optional[int] = None,
            error: Optional[str] = "add",
            random_errors: Optional[Any] = None,
            random_state: Optional[int] = None,
            init_kwargs: Optional[dict[str, Any]] = None,
            **fit_kwargs,
    ):
        super().__init__(
            trend=MODEL_MODE[trend],
            damped=damped,
            seasonal=SEASONALITY_MODE[seasonal],
            seasonal_periods=seasonal_periods,
            error=error,
            random_errors=random_errors,
            random_state=random_state,
            kwargs=init_kwargs,
            **fit_kwargs
        )

    def predict(
            self,
            n: int,
            series=None,
            num_samples: int = 1,
            verbose: Optional[bool] = None,
            show_warnings: bool = True,
            random_state: Optional[int] = None,
    ):
        return super().predict(n=n,num_samples=num_samples,
                               verbose=verbose,show_warnings=show_warnings,random_state=random_state)
# end


class ExponentialSmoothing(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

    def __init__(
            self,
            trend: Optional[str] = "additive",
            damped: Optional[bool] = False,
            seasonal: Optional[str] = "additive",
            seasonal_periods: Optional[int] = None,
            error: Optional[str] = "add",
            random_errors: Optional[Any] = None,
            random_state: Optional[int] = None,
            init_kwargs: Optional[dict[str, Any]] = None,
            # --
            scaler=None,
            # --
            **kwargs,
    ):
        super().__init__(_ExponentialSmoothing, locals())
