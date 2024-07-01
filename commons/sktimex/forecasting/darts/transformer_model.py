import darts.models.forecasting.transformer_model as dm

from .base import BaseDartsForecaster


class TransformerModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation='relu',
        norm_type=None,
        custom_encoder=None,
        custom_decoder=None,
        **kwargs
    ):
        super().__init__(dm.TransformerModel, locals())
        pass
