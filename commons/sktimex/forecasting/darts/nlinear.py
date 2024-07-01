import darts.models.forecasting.nlinear as dm

from .base import BaseDartsForecaster


class NLinearModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        shared_weights=False,
        const_init=True,
        normalize=False,
        use_static_covariates=True,
        **kwargs
    ):
        super().__init__(dm.NLinearModel, locals())
        pass
