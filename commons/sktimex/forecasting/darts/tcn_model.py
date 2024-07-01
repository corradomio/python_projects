import darts.models.forecasting.tcn_model as dm

from .base import BaseDartsForecaster


class TCNModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        kernel_size=3,
        num_filters=3,
        num_layers=None,
        dilation_base=2,
        weight_norm=False,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(dm.TCNModel, locals())
        pass
