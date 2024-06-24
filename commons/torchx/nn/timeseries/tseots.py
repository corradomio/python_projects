#
# Transformers for Time Series
#

__all__ = [
    "TSEncoderOnlyTransformer",
    "TSCNNEncoderTransformer"
]

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .ts import TimeSeriesModel
from .tstran import positional_encoding
from ... import nn as nnx


# ---------------------------------------------------------------------------
# TSEncoderOnlyTransformer
# ---------------------------------------------------------------------------
# Used: EncoderOnlyTransformer
#   Linear projection input features
#   Linear projection output features
#

class TSEncoderOnlyTransformer(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 d_model: int=64,   # alias for feature_size
                 nhead: int=1,
                 num_encoder_layers: int=1,
                 dim_feedforward=None,
                 dropout: float=0.1,
                 device=None, dtype=None,
                 # extended parameters
                 layer_norm=True,
                 positional_encode=True,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, layer_norm=layer_norm,
                         positional_encode=positional_encode,
                         device=device, dtype=dtype, **kwargs)

        input_length, input_size = input_shape
        output_length, output_size = output_shape

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.sqrt_D = torch.tensor(math.sqrt(d_model))

        # to convert input_size into d_model IT IS ENOUGH to use 'Linear(input_size, d_model)'
        # it will be responsibility of the broadcasting mechanism to apply the transformation to
        # ALL elements of the sequence.
        # Note: the data dimension is the LAST dimension, the broadcasting mechanism apply everything
        # following the PREVIOUS dimensions
        self.input_adapter = nnx.Linear(input_size, d_model)

        # self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None
        self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None

        self.dropout = nn.Dropout(dropout)

        self.transformer = nnx.EncoderOnlyTransformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, layer_norm=layer_norm
        )

        self.output_adapter = nnx.Linear(d_model, output_size)

        self.reset_parameters()
        pass
    # end

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # B, S, D = x.shape
        t = self.input_adapter(x)
        t *= self.sqrt_D
        t = t if self.pos_encoding is None else (t + self.pos_encoding)
        t = self.dropout(t)
        t = self.transformer(t, mask=mask)
        t = self.output_adapter(t)
        return t
# end

# ---------------------------------------------------------------------------
# TSEncoderEncoderTransformer (ex TSTransformerV3)
# ---------------------------------------------------------------------------
# Used: CNNEncoderTransformer
#   linear layers replaced by Conv1d layers
#

class TSCNNEncoderTransformer(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 d_model: int=64,   # alias for feature_size
                 nhead: int=1,
                 num_encoder_layers: int=1,
                 dim_feedforward=None,
                 dropout: float=0.1,
                 # extended parameters
                 layer_norm=True,
                 positional_encode=True,
                 average_pooling=False,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, layer_norm=layer_norm,
                         positional_encode=positional_encode,
                         device=device, dtype=dtype, **kwargs)

        input_length, input_size = input_shape
        output_length, output_size = output_shape

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.sqrt_D = torch.tensor(math.sqrt(d_model))

        # to convert input_size into d_model IT IS ENOUGH to use 'Linear(input_size, d_model)'
        # it will be responsibility of the broadcasting mechanism to apply the transformation to
        # ALL elements of the sequence.
        # Note: the data dimension is the LAST dimension, the broadcasting mechanism apply everything
        # following the PREVIOUS dimensions
        self.input_adapter = nnx.Linear(input_size, d_model)

        # self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None
        self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None

        self.transformer = nnx.CNNEncoderTransformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, layer_norm=layer_norm
        )

        self.dropout = nn.Dropout(dropout)

        self.output_adapter = nnx.Linear(d_model, output_size)

        self.reset_parameters()
        pass
    # end

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # B, S, D = x.shape
        t = self.input_adapter(x)
        t *= self.sqrt_D
        t = t if self.pos_encoding is None else (t + self.pos_encoding)
        t = self.dropout(t)
        t = self.transformer(t, mask=mask)
        t = self.output_adapter(t)
        return t
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
