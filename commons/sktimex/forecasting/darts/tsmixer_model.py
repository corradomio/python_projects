import darts.models.forecasting.tsmixer_model as dm

from .base import BaseDartsForecaster


class TSMixerModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        hidden_size=64,
        ff_size=64,
        num_blocks=2,
        activation='ReLU',
        dropout=0.1,
        norm_type='LayerNorm',
        normalize_before=False,
        use_static_covariates=True,
        **kwargs
    ):
        super().__init__(dm.TSMixerModel, locals())
        pass
