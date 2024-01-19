from typing import Union, Callable, Optional, Any

import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

#
# Original 'pytorch transformer file' copied here for debugging.
# See the documentation at the begin of the file
#
# from . import transformer_torch as ttx

# This import refers the 'original' pytorch transformer
from torch.nn.modules import transformer as ttx


# ---------------------------------------------------------------------------
# (Extended) Transformer
# ---------------------------------------------------------------------------

class Transformer(ttx.Transformer):
    """
    It extends the original Pytorch Transformer with flags to specify if to create
    or not the encoder/decoder layer norms. It is reasonable to have the layer norm
    at the both sides (encoder/decoder) or in none. It has no sense to have the normalization
    at one side but not at the other.
    """

    # main configuration parameters
    #   d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
    #   num_encoder_layers: int = 6, num_decoder_layers: int = 6,
    #
    # extended parameters
    #   layer_norm=True
    #
    # Note:
    #   'bias=False' generates several problems
    #

    def __init__(self,
                 # -- main configuration parameters
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 # -- extra parameters
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, device=None, dtype=None,
                 # -- extended parameters
                 layer_norm=True):
        assert bias, "'bias=False' generates several problems"

        factory_kwargs = {'device': device, 'dtype': dtype}
        if not layer_norm:
            encoder_layer = ttx.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first,
                bias, **factory_kwargs)
            custom_encoder = ttx.TransformerEncoder(encoder_layer, num_encoder_layers, None)
        if not layer_norm:
            decoder_layer = ttx.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first,
                bias, **factory_kwargs)
            custom_decoder = ttx.TransformerDecoder(decoder_layer, num_decoder_layers, None)
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


# ---------------------------------------------------------------------------
# EncoderOnlyTransformer
# ---------------------------------------------------------------------------

class MemoryLayer(nn.Module):
    """
    Used as custom decoder to return the encoder output, passed to the 'forward'
    method as second parameter (parameter 'memory')
    """
    def __init__(self):
        super().__init__()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: Optional[bool] = None, memory_is_causal: bool = False):
        # Note: 'memory' is the output of the encoder layer passed as
        # second parameter to the 'decoder' layer
        return memory
# end


class EncoderOnlyTransformer(Transformer):
    """
    Transformer 'encoder only'.
    The 'decoder' ('MemoryLayer') is a custom layer to return just the encoder output
    """
    def __init__(self,
                 # -- main configuration parameters
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6,
                 # [REMOVED] num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 # -- extra parameters
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None,
                 # [REMOVED] custom_decoder = None
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, device=None, dtype=None,
                 # -- extended parameters
                 layer_norm=True):
        super().__init__(d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers, num_decoder_layers=0,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation,
                         layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                         bias=bias, device=device, dtype=dtype,
                         layer_norm=layer_norm,
                         custom_encoder=custom_encoder, custom_decoder=MemoryLayer())
    # end

    def forward(self, src: Tensor, mask=None, **kwargs) -> Tensor:
        # Using the 'MemoryLayer' as custom_decoder, it is not necessary to change the implementation
        # of the previous 'forward(...)' method.
        # However, it is necessary to pass 'tgt'. Simple workaroud: it is used 'src' because it will be used
        # by MemoryLayer to return the encoder output
        return super().forward(src, src, memory_mask=mask, **kwargs)
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
