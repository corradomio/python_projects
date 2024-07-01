import darts.models.forecasting.nbeats as dm

from .base import BaseDartsForecaster


class NBEATSModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        generic_architecture=True,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
        expansion_coefficient_dim=5,
        trend_polynomial_degree=2,
        dropout=0.0,
        activation='ReLU',
        **kwargs
    ):
        super().__init__(dm.NBEATSModel, locals())
        pass
