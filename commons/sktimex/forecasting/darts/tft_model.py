import darts.models.forecasting.tft_model as dm

from .base import BaseDartsForecaster


class TFTModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        full_attention=False,
        feed_forward='GatedResidualNetwork',
        dropout=0.1,
        hidden_continuous_size=8,
        categorical_embedding_sizes=None,
        add_relative_index=False,
        loss_fn=None,
        likelihood=None,
        norm_type='LayerNorm',
        use_static_covariates=True,
        **kwargs
    ):
        super().__init__(dm.TFTModel, locals())
        pass
