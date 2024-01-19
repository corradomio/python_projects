#
# Transformers for Time Series
#
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import math

from torch import Tensor
from .ts import TimeSeriesModel
from ... import nn as nnx

__all__ = [
    "TSTransformerV1",
    "TSTransformerV2",
    "TSTransformerV3"
]


# ---------------------------------------------------------------------------
# PositionalReplicate
# ---------------------------------------------------------------------------

class PositionalReplicate(nn.Module):
    def __init__(self, n_repeat=1, input_size=0, ones=False):
        """
        Repeat 'r' times a 3D tensor (batch, seq, input) along 'input' dimension, generating a new 3D tensor
        with shape (batch, seq, r*input).

        The tensor can be 'expanded' (along 'input' dimension) if 'input_size' is not zero and it is not equals
        to the current input size. If 'input_size' is > 0, some zeroes are added in front, otherwise at the back


        :param n_repeat: n of times the tensor is repeated along the 'input' dimension
        :param input_size: if to add extra zeros in front (input_size > 0) or at back (input_size < 0)
            of the tensor, to made it with exactly 'input_size' features
        :param ones: if to use 1 or 0 during the expansion 'normalize' the vector.
        """
        super().__init__()
        self.n_repeat = n_repeat
        self.input_size = input_size
        self.ones = ones
        assert n_repeat > 0, "'n_repeat' needs to be an integer > 0"
        assert isinstance(input_size, int), "'input_size' needs to be an integer"

    def forward(self, x: Tensor) -> Tensor:
        n_repeat = self.n_repeat
        data_size = x.shape[-1]
        input_size = self.input_size
        const = torch.ones if self.ones else torch.zeros

        if input_size != 0 and data_size != abs(input_size):
            expand = input_size + data_size if input_size < 0 else input_size - data_size
        else:
            expand = 0

        if expand > 0:
            # expand adding 0/1 in front: [0..., x]
            shape = list(x.shape)
            shape[2] += expand
            z = const(shape)
            z[:, :, expand:] = x
            x = z
        elif expand < 0:
            # expand adding 0/1 at back: [x, 0...]
            shape = list(x.shape)
            shape[2] -= expand
            z = const(shape)
            z[:, :, :expand] = x
            x = z

        if n_repeat > 1:
            x = x.repeat(1, 1, n_repeat)

        return x
# end


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def positional_encoding(seq_len, d, n=10000, dtype=torch.float32, astensor=True):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i+0] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return torch.tensor(P[np.newaxis, ...], dtype=dtype) if astensor else P


class PositionalEncoder(nn.Module):

    def __init__(self, in_features, max_len=1000, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.max_len = max_len

        # self.pos = torch.zeros((1, max_len, in_features), dtype=dtype)
        # X = (torch.arange(max_len, dtype=dtype).reshape(-1, 1) /
        #      torch.pow(10000, torch.arange(0, in_features, 2, dtype=dtype) / in_features))
        # self.pos[0, :, 0::2] = torch.sin(X)
        # self.pos[0, :, 1::2] = torch.cos(X)

        self.pos = positional_encoding(max_len, in_features, dtype=dtype)

    def forward(self, input):
        seq_len = input.shape[1]
        t = input + self.pos[0, 0:seq_len, :].to(input.device)
        return t
# end


# ---------------------------------------------------------------------------
# TSTransformerV1
# ---------------------------------------------------------------------------
# Input features extended & replicated
#

class TSTransformerV1(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 nhead=1,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=None,
                 dropout=0,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, **kwargs)
        input_length, input_size = input_shape
        output_length, output_size = output_shape
        d_model = nhead*input_size

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.replicate = PositionalReplicate(
            nhead, input_size
        )
        self.decoder_output_adapter = nnx.TimeDistributed(
            nnx.Linear(d_model, output_size)
        )

        self.transformer = nnx.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            **kwargs
        )

        max_len = max(input_shape[0], output_shape[0])
        self.positional_encoder = PositionalEncoder(d_model, max_len)
        pass
    # end

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return self._forward_train(x)
        else:
            return self._forward_predict(x)

    def _forward_train(self, x):
        x_enc, x_dec = x

        x_enc = self.replicate(x_enc)
        x_dec = self.replicate(x_dec)

        x_enc = self.positional_encoder(x_enc)
        x_dec = self.positional_encoder(x_dec)

        y_tran = self.transformer(x_enc, x_dec)

        yp = self.decoder_output_adapter(y_tran)
        return yp
    # end

    def _forward_predict(self, x):
        output_seqlen, output_size = self.output_shape

        x_enc = x                           # [N, Lin, Hin]
        x_dec = x[:, -1:, -output_size:]    # [N, 1,  Hout]

        x_enc = self.replicate(x_enc)
        x_enc = self.positional_encoder(x_enc)

        y_enc = self.transformer.encoder(x_enc)

        ylist = []
        for i in range(output_seqlen):
            x_dec = self.replicate(x_dec)
            x_dec = self.positional_encoder(x_dec)

            y_pred = self.transformer.decoder(x_dec, y_enc)
            y_pred = self.decoder_output_adapter(y_pred)
            ylist.append(y_pred)

            x_dec = y_pred
        # end
        return torch.cat(ylist, dim=1)
    # end
# end


# ---------------------------------------------------------------------------
# TSTransformerV2
# ---------------------------------------------------------------------------
# Input features expanded using a single matrix applied to all elements in the
# sequence.

class TSTransformerV2(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 d_model=64,
                 nhead=1,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 layer_norm=True,
                 dim_feedforward=None,
                 dropout=0,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         layer_norm=layer_norm,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, **kwargs)
        input_length, input_size = input_shape
        output_length, output_size = output_shape

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.encoder_input_adapter = nnx.TimeDistributed(
            nnx.Linear(input_size, d_model)
        )
        self.decoder_input_adapter = nnx.TimeDistributed(
            nnx.Linear(output_size, d_model)
        )
        self.decoder_output_adapter = nnx.TimeDistributed(
            nnx.Linear(d_model, output_size)
        )

        self.tran = nnx.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm=layer_norm,
            **kwargs
        )
        pass
    # end

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return self._forward_train(x)
        else:
            return self._forward_predict(x)
    # end

    def _forward_train(self, x):
        x_enc, x_dec = x

        x_enc = self.encoder_input_adapter(x_enc)
        x_dec = self.decoder_input_adapter(x_dec)

        y_tran = self.tran(x_enc, x_dec)

        yp = self.decoder_output_adapter(y_tran)
        return yp
    # end

    def _forward_predict(self, x):
        output_seqlen, output_size = self.output_shape

        x_enc = x                           # [N, Lin, Hin]
        x_dec = x[:, -1:, -output_size:]    # [N, 1,  Hout]

        x_enc = self.encoder_input_adapter(x_enc)
        y_enc = self.tran.encoder(x_enc)

        ylist = []
        for i in range(output_seqlen):
            x_dec = self.decoder_input_adapter(x_dec)
            y_pred = self.tran.decoder(x_dec, y_enc)
            y_pred = self.decoder_output_adapter(y_pred)

            ylist.append(y_pred)
            x_dec = y_pred
        # end
        return torch.cat(ylist, dim=1)
    # end
# end


# ---------------------------------------------------------------------------
# TSTransformerV3
# ---------------------------------------------------------------------------
# Used: EncoderOnlyTransformer
#   Linear projection input features
#   Linear projection output features
#

class TSTransformerV3(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 d_model: int=64,
                 nhead: int=1,
                 num_encoder_layers: int=1,
                 dim_feedforward=None,
                 dropout: float=0.1,
                 layer_norm=True,
                 positional_encode=True,
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
        self.input_projection = nnx.Linear(input_size, d_model)

        # self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None
        self.pos_encoding = positional_encoding(input_length, d_model) if positional_encode else None

        self.dropout = nn.Dropout(dropout)

        self.transformer = nnx.EncoderOnlyTransformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, layer_norm=layer_norm
        )

        self.output_projection = nnx.Linear(d_model, output_size)

        self.reset_parameters()
        pass
    # end

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # B, S, D = x.shape
        t = self.input_projection(x)
        t *= self.sqrt_D
        t = t if self.pos_encoding is None else (t + self.pos_encoding)
        t = self.dropout(t)
        t = self.transformer(t, mask=mask)
        t = self.output_projection(t)
        return t

# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
