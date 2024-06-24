
__all__ = [
    "DartsLinearForecaster"
]

from typing import Optional, Tuple, Union, List, Dict

import darts.models.forecasting.linear_regression_model

from .base import BaseDartsForecaster

LAGS_TYPE = Union[int, List[int], Dict[str, Union[int, List[int]]]]
FUTURE_LAGS_TYPE = Union[
    Tuple[int, int], List[int], Dict[str, Union[Tuple[int, int], List[int]]]
]


# ---------------------------------------------------------------------------

class DartsLinearForecaster(BaseDartsForecaster):

    def __init__(
        self, *,
        lags: Optional[LAGS_TYPE] = None,
        lags_past_covariates: Optional[LAGS_TYPE] = None,
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = None,
        quantiles: Optional[List[float]] = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        **kwargs
    ):
        kwargs = dict(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            likelihood=likelihood,
            quantiles=quantiles,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
        ) | kwargs
        super().__init__(
            darts.models.forecasting.linear_regression_model.LinearRegressionModel,
            kwargs
        )
        pass

    def _compose_kwargs(self, y, X=None, fh=None):
        kwargs = self.kwargs
        if X is not None:
            kwargs = kwargs | {'lags_past_covariates': kwargs['lags']}
        if fh is not None:
            kwargs = kwargs | {'lags_future_covariates': len(fh)}
        return kwargs
