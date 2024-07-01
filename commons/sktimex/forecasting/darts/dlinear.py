import darts.models.forecasting.dlinear as dm

from .base import BaseDartsForecaster


class DLinearModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        shared_weights=False,
        kernel_size=25,
        const_init=True,
        use_static_covariates=True,
        **kwargs
    ):
        super().__init__(dm.DLinearModel, locals())
        pass
