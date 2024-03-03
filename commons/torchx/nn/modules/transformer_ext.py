from typing import Union, Callable, Optional, Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# This import refers the 'original' pytorch transformer
from torch.nn.modules import transformer as ttx

# extended Conv1d layer
from ..modules import cnn as cnnx


#
# Original 'pytorch transformer file' copied here as
# 'transformer_torch', used for debugging.
# See the documentation at the begin of the file
#
# from . import transformer_torch as ttx


# ---------------------------------------------------------------------------
# (Extended) Transformer
# ---------------------------------------------------------------------------

class Transformer(ttx.Transformer):
    """
    It extends the original Pytorch Transformer with flags to specify if to create
    or not the encoder/decoder layer norms. It is reasonable to have the layer norm
    at the both sides (encoder/decoder) or in none.
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

    def forward(self, src: Tensor, mask=None, is_causal=False, **kwargs) -> Tensor:
        # Using 'MemoryLayer' as custom_decoder, it is not necessary to change the implementation
        # of the previous 'forward(...)' method.
        # However, it is necessary to pass 'tgt'. Simple workaroud: it is used 'src' because it will be used
        # by MemoryLayer to return the encoder output
        return super().forward(src, src, memory_mask=mask,
                               src_is_causal=is_causal, memory_is_causal=is_causal,
                               **kwargs)
    # end
# end


# ---------------------------------------------------------------------------
# CNNEncoderTransformer
# ---------------------------------------------------------------------------
# Custom encoder where the linear layers are replaced by 
#
# # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs
#
#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

class LayerNoNorm(nn.Module):
    def __init__(self, eps=1.0e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        return input
# end


class CNNEncoderLayer(ttx.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 # extended parameter
                 layer_norm=True,
                 # specific for Conv1d
                 kernel_size: int = 1, stride: int = 1, padding: str = "same"
                 ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias, device=device, dtype=dtype,
        )
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Check if to disable the normalization layers
        # Trick: to force 'why_not_sparsity_fast_path' to be NOT an empty string
        # the most simple way is to have different 'eps' values in 'norm1' and 'norm2'
        # For the reason to see the notes in 'CNNEncoderTransformer.forward'
        if not layer_norm:
            self.norm1 = LayerNoNorm(eps=layer_norm_eps)
            self.norm2 = LayerNoNorm(eps=1.5*layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=1.5*layer_norm_eps, bias=bias, **factory_kwargs)

        # replace the linear layers with Conv1d layers
        self.linear1 = cnnx.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        self.linear2 = cnnx.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
    # end

    # Used to rename 'mask' into 'src_mask'
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False) -> Tensor:

        return super().forward(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

    # [DEBUG] for debugging ONLY
    # this is copy/paste of original 'TransformerEncoderLayer' method
    # reorganized to simplify the debugging
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)     # now it is a Conv1d layer
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)     # now it is a Conv1d layer
        x = self.dropout2(x)
        return x
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # return self.dropout2(x)
# end


class CNNEncoderTransformer(Transformer):
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
                 # [REMOVED]
                 # custom_encoder: Optional[Any] = None,
                 # custom_decoder = None
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, device=None, dtype=None,
                 # -- extended parameters
                 layer_norm=True,
                 # -- specific for Conv1d
                 kernel_size: int = 1, stride: int = 1, padding: str = "same"

    ):
        super().__init__(d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers, num_decoder_layers=0,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation,
                         layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                         bias=bias, device=device, dtype=dtype,
                         layer_norm=layer_norm,
                         custom_encoder=CNNEncoderLayer(
                             d_model=d_model,
                             nhead=nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation,
                             layer_norm_eps=layer_norm_eps,
                             batch_first=batch_first,
                             norm_first=norm_first,
                             bias=bias, device=device, dtype=dtype,
                             # extended parameters
                             layer_norm=layer_norm,
                             # specific for Conv1d
                             kernel_size=kernel_size, stride=stride, padding=padding
                         ),
                         custom_decoder=MemoryLayer())
    # end

    def forward(self, src: Tensor, mask=None, is_causal=False, **kwargs) -> Tensor:
        # Using 'MemoryLayer' as custom_decoder, it is not necessary to change the implementation
        # of the previous 'forward(...)' method.
        # However, it is necessary to pass 'tgt'. Simple workaroud: it is used 'src' because it will be used
        # by MemoryLayer to return the encoder output

        # Problem: to use CNN it is necessary to force 'why_not_sparsity_fast_path' to a string
        # in original 'Transformer.forward()' implementation.
        # There are several methods, the most simple one us to use little different values of
        # 'eps' in 'norm1' and 'norm2'
        # This is necessary because in 'Transformer.forward()' implementation there are checks on
        # the weight dimensions in 'self.linear1' and 'self.linear2' BUT now these layers are 'Conv1d'
        # layers!!!
        return super().forward(src, src, memory_mask=mask,
                               src_is_causal=is_causal, memory_is_causal=is_causal,
                               **kwargs)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
