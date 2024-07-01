import darts.models.forecasting.rnn_model as dm

from .base import BaseDartsForecaster


class RNNModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        model='RNN',
        hidden_dim=25,
        n_rnn_layers=1,
        dropout=0.0,
        training_length=24,
        **kwargs
    ):
        super().__init__(dm.RNNModel, locals())
        pass
