import darts.models.forecasting.nhits as dm

from .base import BaseDartsForecaster


class NHiTSModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        pooling_kernel_sizes=None,
        n_freq_downsample=None,
        dropout=0.1,
        activation='ReLU',
        MaxPool1d=True,
        **kwargs
    ):
        super().__init__(dm.NHiTSModel, locals())
        pass
