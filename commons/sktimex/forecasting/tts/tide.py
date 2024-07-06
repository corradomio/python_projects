from .base import BaseTTSForecaster
import torchx.nn.timeseries.tide as tts


class TiDE(BaseTTSForecaster):

    _tags = {
        'future-features': True
    }

    def __init__(
        self,
        lags=12,
        tlags=1,
        hidden_size=None,
        decoder_output_size=None,
        temporal_hidden_size=None,
        num_encoder_layers=1,
        num_decoder_layers=1,
        use_layer_norm=True,
        use_future_features=True,
        dropout=0.1,

        engine_kwargs={}
    ):
        super().__init__(tts.TiDE, locals())
        pass