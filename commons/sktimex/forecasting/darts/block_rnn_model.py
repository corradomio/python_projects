import darts.models.forecasting.block_rnn_model as dm

from .base import BaseDartsForecaster


class BlockRNNModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        model='RNN',
        hidden_dim=25,
        n_rnn_layers=1,
        hidden_fc_sizes=None,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(dm.BlockRNNModel, locals())
        pass
