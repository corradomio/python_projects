#
# Transformers for Time Series
#

import torch

from .ts import TimeSeriesModel
from .tspos import positional_encoding
from .tsutils import apply_if
from ... import nn as nnx

__all__ = [
    "TSPlainTransformer",
]


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

# WARNING / Trick:
#   it is necessary to understand IF the model is called during the
#   train step OR the prediction step.
#   The flag 'self.training' is not enough BECAUSE it can be False ALSO
#   during the training.
#   The problem is this: during the 'real prediction', the model must be
#   used in 'recursive' way. It is necessary to pass ALSO the X_decoder
#   data. BUT this data is not all available, then, for default, is filled
#   with 0
#

class TSPlainTransformer(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 d_model: int=64,   # alias for feature_size
                 nhead: int=1,
                 num_encoder_layers: int=1,
                 num_decoder_layers: int=1,
                 layer_norm=True,
                 dim_feedforward=None,
                 decoder_offset: int=-1,
                 dropout: float=0.1,
                 positional_encode=True,
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         layer_norm=layer_norm,
                         dim_feedforward=dim_feedforward,
                         decoder_offset=-1,
                         dropout=dropout,
                         positional_encode=positional_encode, **kwargs)
        self.decoder_offset = decoder_offset

        input_length, input_size = input_shape
        output_length, output_size = output_shape

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.encoder_input_adapter = None
        self.decoder_input_adapter = None
        self.decoder_output_adapter = None
        self.pos_encoding = None

        if input_size != d_model:
            self.encoder_input_adapter = nnx.Linear(input_size, d_model)
        if d_model != output_size:
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
        if positional_encode:
            pos_length = max(input_length, output_length)
            self.pos_encoding = positional_encoding(pos_length, d_model) if positional_encode else None
    # end

    def forward(self, x, **kwargs):
        self._check_decoder_input_adapter(x)

        # doffset = kwval(kwargs, 'decoder_offset', None)
        # if doffset in [None, "pass", 0]:
        #     return self._train_forward(x)
        # else:
        #     return self._predict_forward(x, doffset)

        # doffset = kwval(kwargs, 'decoder_offset', None)
        # recursive = kwval(kwargs, 'recursive', False)
        #
        # if doffset is None and not recursive:
        #     return self._train_forward(x)
        # else:
        #     return self._predict_forward(x, self.decoder_offset)

        if not self._is_model_prediction(x):
            return self._train_forward(x)
        else:
            return self._predict_forward(x, self.decoder_offset)
    # end

    def _is_model_prediction(self, x):
        assert isinstance(x, (list, tuple))
        doffset = self.decoder_offset
        output_size = self.output_shape[1]

        x_dec = x[1]
        dlen = x_dec.shape[1]
        for i in range(-doffset, dlen):
            for j in range(output_size):
                if x_dec[0, i, j] != 0:
                    return False
        return True

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

        # check validity
        doffset = self.decoder_offset
        assert x_enc[0, doffset, 0] == x_dec[0, 0, 0]

        x_enc_a = apply_if(x_enc, self.encoder_input_adapter)
        x_dec_a = apply_if(x_dec, self.decoder_input_adapter)

        if self.pos_encoding is not None:
            x_enc_a += self.pos_encoding[:x_enc_a.shape[1], :]
            x_dec_a += self.pos_encoding[:x_dec_a.shape[1], :]

        yp_a = self.tran(x_enc_a, x_dec_a)
        yp = apply_if(yp_a, self.decoder_output_adapter)

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

        # prepare the placeholder 'y_pred'
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
            yp = apply_if(yp_a, self.decoder_output_adapter)

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
