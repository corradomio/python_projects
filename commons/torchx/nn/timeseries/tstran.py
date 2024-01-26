#
# Transformers for Time Series
#

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from stdlib import kwval
from .ts import TimeSeriesModel
from ... import nn as nnx

__all__ = [
    "TSPlainTransformer",
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
# end


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
# TSTransformerWithReplicate (ex TSTransformerV1)
# ---------------------------------------------------------------------------
# Input features extended & replicated
#

# class TSTransformerWithReplicate(TimeSeriesModel):
#     def __init__(self, input_shape, output_shape,
#                  nhead=1,
#                  num_encoder_layers=1,
#                  num_decoder_layers=1,
#                  dim_feedforward=None,
#                  dropout=0,
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          nhead=nhead,
#                          num_encoder_layers=num_encoder_layers,
#                          num_decoder_layers=num_decoder_layers,
#                          dim_feedforward=dim_feedforward,
#                          dropout=dropout, **kwargs)
#         input_length, input_size = input_shape
#         output_length, output_size = output_shape
#         d_model = nhead*input_size
#
#         if dim_feedforward in [0, None]:
#             dim_feedforward = d_model
#
#         self.replicate = PositionalReplicate(
#             nhead, input_size
#         )
#         self.decoder_output_adapter = nnx.TimeDistributed(
#             nnx.Linear(d_model, output_size)
#         )
#
#         self.transformer = nnx.Transformer(
#             d_model=d_model, nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             **kwargs
#         )
#
#         max_len = max(input_shape[0], output_shape[0])
#         self.positional_encoder = PositionalEncoder(d_model, max_len)
#         pass
#     # end
#
#     def forward(self, x):
#         if isinstance(x, (list, tuple)):
#             return self._forward_train(x)
#         else:
#             return self._forward_predict(x)
#
#     def _forward_train(self, x):
#         x_enc, x_dec = x
#
#         x_enc = self.replicate(x_enc)
#         x_dec = self.replicate(x_dec)
#
#         x_enc = self.positional_encoder(x_enc)
#         x_dec = self.positional_encoder(x_dec)
#
#         y_tran = self.transformer(x_enc, x_dec)
#
#         yp = self.decoder_output_adapter(y_tran)
#         return yp
#     # end
#
#     def _forward_predict(self, x):
#         output_seqlen, output_size = self.output_shape
#
#         x_enc = x                           # [N, Lin, Hin]
#         x_dec = x[:, -1:, -output_size:]    # [N, 1,  Hout]
#
#         x_enc = self.replicate(x_enc)
#         x_enc = self.positional_encoder(x_enc)
#
#         y_enc = self.transformer.encoder(x_enc)
#
#         ylist = []
#         for i in range(output_seqlen):
#             x_dec = self.replicate(x_dec)
#             x_dec = self.positional_encoder(x_dec)
#
#             y_pred = self.transformer.decoder(x_dec, y_enc)
#             y_pred = self.decoder_output_adapter(y_pred)
#             ylist.append(y_pred)
#
#             x_dec = y_pred
#         # end
#         return torch.cat(ylist, dim=1)
#     # end
# # end


# ---------------------------------------------------------------------------
# TSPlainTransformer (ex TSTransformerV2)
# ---------------------------------------------------------------------------
# Input features expanded using a single matrix applied to all elements in the
# sequence.
# Training using 2 inputs:
#   1) input  for encoder       [X|y] past window
#   2) input  for the decoder   [X|y] (forecasting window - 1)
#   3) output for the decoder   [y] forecasting window
#

class TSPlainTransformer(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 d_model=64,
                 nhead=1,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 layer_norm=True,
                 dim_feedforward=None,
                 dropout=0,
                 positional_encode=True,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         layer_norm=layer_norm,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         positional_encode=positional_encode, **kwargs)
        input_length, input_size = input_shape
        output_length, output_size = output_shape

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.encoder_input_adapter = nnx.Linear(input_size, d_model)
        self.decoder_input_adapter = None
        self.decoder_output_adapter = nnx.Linear(d_model, output_size)

        self.tran = nnx.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm=layer_norm,
            **kwargs
        )

        # in theory 'input_length > output_length' BUT to be safe, it is better to use 'max'
        pos_length = max(input_length, output_length)
        self.pos_encoding = positional_encoding(pos_length, d_model) if positional_encode else None
    # end

    def forward(self, x, **kwargs):
        self._check_decoder_input_adapter(x)

        doffset = kwval(kwargs, ('decoder', 'doffset'), None)
        if doffset in [None, "pass", 0]:
            return self._train_forward(x)
        else:
            return self._predict_forward(x, doffset)
    # end

    def _check_decoder_input_adapter(self, x):
        # IF the decoder's input has a size different than the encoder's input
        # it is necessary to use a different adapter, otherwise it is possible
        # to use the same one
        if not self.training or self.decoder_input_adapter is not None:
            return

        x_enc, x_dec = x
        if x_enc.shape[-1] == x_dec.shape[-1]:
            self.decoder_input_adapter = self.encoder_input_adapter
        else:
            self.decoder_input_adapter = nnx.Linear(x_dec.shape[-1], self.d_model)
    # end

    def _train_forward(self, x):
        x_enc, x_dec = x
        x_enc_a = self.encoder_input_adapter(x_enc)
        x_dec_a = self.decoder_input_adapter(x_dec)

        if self.pos_encoding is not None:
            x_enc_a += self.pos_encoding[:x_enc_a.shape[1], :]
            x_dec_a += self.pos_encoding[:x_dec_a.shape[1], :]

        yp_a = self.tran(x_enc_a, x_dec_a)
        yp = self.decoder_output_adapter(yp_a)

        return yp
    # end

    def _predict_forward(self, x, doffset):
        # It predict |decoder_offset| targets at each interaction
        assert isinstance(doffset, int) and doffset < 0
        output_seqlen, output_size = self.output_shape
        dlen = -doffset

        # extract X's components
        x_enc, x_dec_seq = x
        batch = len(x_enc)

        # prepare 'y_pred'
        y_pred_shape = (batch, ) + self.output_shape
        y_pred = torch.zeros(y_pred_shape, dtype=x_enc.dtype)

        x_enc_a = self.encoder_input_adapter(x_enc)
        if self.pos_encoding is not None:
            enc_len = x_enc_a.shape[1]
            x_enc_a += self.pos_encoding[:enc_len, :]

        i = 0
        iend = min(i + dlen, output_seqlen)
        while i < output_seqlen:
            x_dec = x_dec_seq[:, i:iend]

            x_dec_a = self.decoder_input_adapter(x_dec)
            if self.pos_encoding is not None:
                dec_len = x_dec_a.shape[1]
                x_dec_a += self.pos_encoding[:dec_len, :]

            yp_a = self.tran(x_enc_a, x_dec_a)
            yp = self.decoder_output_adapter(yp_a)

            # fill y_pred
            y_pred[:, i:iend, :] = yp

            # advance of dlen
            # Note: advance `i` HERE because in this way, it is possible to save yp in the
            #       NEXT position in 'x_dec_seq'
            i += dlen
            iend = min(i + dlen, output_seqlen)

            # fill x_dec_temp BUT ONLY the necessary slots
            if i < iend:
                x_dec_seq[:, i:iend, :output_size] = yp
        # end

        # done
        return y_pred
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
