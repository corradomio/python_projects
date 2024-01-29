from math import exp
from random import random

import torch

from .ts import TimeSeriesModel
from ... import nn as nnx
from ...utils import time_repeat

__all__ = [
    "TSSeq2SeqV1",
    "TSSeq2SeqV1",
    "TSSeq2SeqV3"
]

# ---------------------------------------------------------------------------
# TSSeq2SeqV1
# ---------------------------------------------------------------------------
# encoder -[hidden_state]-> decoder
#
# the decoder receives in input zeros.

class TSSeq2SeqV1(TimeSeriesModel):
    """
    Simple model:
        [X_train] -> encoder -> [hidden_state] -> decoder -> [y_predict]
                             -> [zero]

    The input of the decoder is the ZERO vector
    """

    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 flavour='lstm', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         **kwargs)

        input_seqlen, input_size = input_shape
        ouput_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size

        enc_params = {} | kwargs
        enc_params['input_size'] = feature_size
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = True

        dec_params = {} | kwargs
        dec_params['input_size'] = hidden_size
        dec_params['hidden_size'] = hidden_size

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        self.input_adapter = None
        self.output_adapter = None

        if feature_size != input_size:
            self.input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)
        if hidden_size != output_size:
            self.output_adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)

        self._zero_cache: dict[int, torch.Tensor] = {}

    def forward(self, x):
        t = self.input_adapter(x) if self.input_adapter is not None else x

        t, h = self.enc(t)
        z = self._zero(len(t), t)
        t = self.dec(z, h)

        y = self.output_adapter(t) if self.output_adapter is not None else t
        return y

    def _zero(self, batch_size, t):
        # t: tensor used to retrieve dtype and device
        if batch_size not in self._zero_cache:
            hidden_size = self.hidden_size
            output_seqlen = self.output_shape[0]
            zero = torch.zeros((batch_size, output_seqlen, hidden_size), dtype=t.dtype, device=t.device, requires_grad=False)
            self._zero_cache[batch_size] = zero
        return self._zero_cache[batch_size]
    # end
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqV2
# ---------------------------------------------------------------------------
# encoder output is used as input into decoder.
# if the encoder's output is just the last value -> it is necessary to replicate
#   it to adapt it to the decoder's input
# if the encoder's output is the complete sequence -> it is necessary to use
#   a linear layer to adapt it to the decoder's input
#

class TSSeq2SeqV2(TimeSeriesModel):
    """
    Simple model:
        [X_train] -> encoder -> [hidden_state] -> decoder -> [y_predict]
                             -> [y_encoder]

    but the encoder output [y_encoder] is compared against [y_train]
    """

    _tags = {
        "x-use-ypredict": False,
        "y-use-ytrain": True
    }

    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 flavour='lstm',
                 use_encoder_sequence=False,
                 **kwargs):
        """

        :param input_shape:
        :param output_shape:
        :param feature_size:
        :param hidden_size:
        :param flavour:
        :param use_encoder_sequence: if to use just the last encoder's value (False)
            or the complete sequence (True)
        :param kwargs:
        """
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         use_encoder_sequence=use_encoder_sequence,
                         **kwargs)

        input_seqlen, input_size = input_shape
        output_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.use_encoder_sequence = use_encoder_sequence

        enc_params = {} | kwargs
        enc_params['input_size'] = feature_size
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = True       # return the hidden state
        enc_params['return_sequence'] = use_encoder_sequence in (True, "return")
            # False return just the last value
            # True  return the encoder sequence

        dec_params = {} | kwargs
        dec_params['input_size'] = feature_size
        dec_params['hidden_size'] = hidden_size
        dec_params['return_state'] = False

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        self.enc_input_adapter = None
        self.dec_input_adapter = None
        self.output_adapter = None

        if feature_size != input_size:
            self.enc_input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)
        if use_encoder_sequence:
            self.dec_input_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=(output_seqlen, feature_size))
        else:
            self.dec_input_adapter = nnx.TimeLinear(in_features=hidden_size, out_features=feature_size, replicate=output_seqlen)
        if feature_size != output_size:
            self.output_adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)
    # end

    def forward(self, x):
        # apply the encoder input adapter
        t = self.enc_input_adapter(x) if self.enc_input_adapter is not None else x

        # the encoder can return just the last value or the complete sequence!
        # 'e': encoder output
        # 'h': encoder state
        e, h = self.enc(t)

        # convert the encoder output into the decoder input
        d = self.dec_input_adapter(e) if self.dec_input_adapter is not None else e

        # 4) call the decoder
        t = self.dec(d, h)

        # 5) apply the decoder output adapter
        y = self.output_adapter(t) if self.output_adapter is not None else t

        # 6) return (encoder_output, decoder_output) OR only decoder_output
        if self.use_encoder_sequence == 'return':
            e = self.output_adapter(e) if self.output_adapter is not None else e
            return e, y
        else:
            return y
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqV3
# ---------------------------------------------------------------------------
# fit((Xt, yp), (yt, yp))
#

class TSSeq2SeqV3(TimeSeriesModel):
    """
    Simple model:
        [X_train] -> encoder -> [hidden_state] -> decoder -> [y_predict]
                             -> [y_encoder]

    but the encoder output [y_encoder] is compared [y_train] AND it is used
    the mechanism 'teacher forcing'
    """

    _tags = {
        "x-use-ypredict": True,
        "y-use-ytrain": True
    }

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm',
                 target_first=True,
                 teacher_forcing=0.01,
                 **kwargs):
        """

        :param input_shape:
        :param output_shape:
        :param hidden_size:
        :param flavour:
        :param target_first: the the target is present in the first (True) or last (False) columns
            of the tensor X used for the training/prediction
        :param teacher_forcing: decay exponent used for the 'teacher forcing' mechanism.
        :param kwargs:
        """
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         hidden_size=hidden_size,
                         target_first=target_first,
                         teacher_forcing=teacher_forcing,
                         **kwargs)

        self.hidden_size = hidden_size
        self.target_first = target_first
        self.teacher_forcing = teacher_forcing

        if hidden_size is None:
            hidden_size = output_shape[1]

        output_size = output_shape[1]

        enc_params = {} | kwargs
        enc_params['input_size'] = input_shape[1]
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = True       # return the hidden state
        enc_params['return_sequence'] = True    # return all encoder sequence

        dec_params = {} | kwargs
        dec_params['input_size'] = output_size
        dec_params['hidden_size'] = hidden_size
        dec_params['return_state'] = True

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        self.adapter = None if hidden_size == output_size \
            else nnx.Linear(in_features=hidden_size, out_features=output_size)

        # n of times 'forward' is called
        self.iteach = 0
    # end

    def forward(self, xytp):
        if isinstance(xytp, (list, tuple)):
            return self._train_forward(xytp)
        else:
            return self._predict_forward(xytp)

    def _train_forward(self, xyp):
        # xyp = (X, y_pred)
        x, yp = xyp
        # n of features in y
        output_seqlen, my = self.output_shape
        self.iteach += 1

        # e: encoder sequence
        # t: decoder sequence
        e, h = self.enc(x)
        e = e if self.adapter is None else self.adapter(e)

        # yprev
        if self.target_first:
            yprev = x[:, -1:, :my]
        else:
            yprev = x[:, -1:, -my:]

        ylist = []
        for i in range(output_seqlen):
            if i > 0 and self.teacher_forcing > 0:
                prob = exp(-self.iteach * self.teacher_forcing)
                if random() < prob:
                    # yprev[:, :, :] = yp[:, i-1:i, :]
                    yprev.data = torch.clone(yp[:, i-1:i, :]).data
                else:
                    pass

            yc, h = self.dec(yprev, h)
            yc = yc if self.adapter is None else self.adapter(yc)
            ylist.append(yc)
            yprev = yc

        t = torch.cat(ylist, dim=1)
        return e, t

    def _predict_forward(self, x):
        e, h = self.enc(x)
        e = e if self.adapter is None else self.adapter(e)
        output_seqlen, my = self.output_shape

        # yprev
        if self.target_first:
            yprev = e[:, -1:, :my]
        else:
            yprev = e[:, -1:, -my:]

        ylist = []
        for i in range(output_seqlen):
            yc, h = self.dec(yprev, h)
            yc = yc if self.adapter is None else self.adapter(yc)
            ylist.append(yc)
            yprev = yc

        t = torch.cat(ylist, dim=1)
        return t
# end
