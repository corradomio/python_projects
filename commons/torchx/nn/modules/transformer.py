from typing import Union, Callable, Optional, Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .transformer_torch import TransformerTorch

nn_Transformer = TransformerTorch


class Transformer(nn_Transformer):

    # main configuration parameters
    #   d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
    #   num_encoder_layers: int = 6, num_decoder_layers: int = 6,
    # extended parameters
    #   encoder_layer_norm=True, decoder_layer_norm=True
    #

    def __init__(self,
                 # main configuration parameters
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 # extra parameters
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = False, device=None, dtype=None,
                 # extended parameters
                 encoder_layer_norm=True, decoder_layer_norm=True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        if not encoder_layer_norm:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first,
                bias, **factory_kwargs)
            custom_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, None)
        if not decoder_layer_norm:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first,
                bias, **factory_kwargs)
            custom_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, None)
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, src: Tensor, tgt: Tensor, **kwargs) -> Tensor:
        return super().forward(src, tgt, **kwargs)
# end
