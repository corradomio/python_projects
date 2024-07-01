import darts.models.forecasting.linear_regression_model as dm

from .base import BaseDartsForecaster


class LinearRegressionModel(BaseDartsForecaster):

    def __init__(
        self, *,
        lags=None,
        lags_past_covariates=None,
        lags_future_covariates=None,
        output_chunk_length=1,
        output_chunk_shift=0,
        add_encoders=None,
        likelihood=None,
        quantiles=None,
        random_state=None,
        multi_models=True,
        use_static_covariates=True,
        **kwargs
    ):
        super().__init__(dm.LinearRegressionModel, locals())
        pass
