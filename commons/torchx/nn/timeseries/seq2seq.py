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
    """

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         hidden_size=hidden_size,
                         **kwargs)

        self.hidden_size = hidden_size

        if hidden_size is None:
            hidden_size = output_shape[1]

        output_size = output_shape[1]

        enc_params = {} | kwargs
        enc_params['input_size'] = input_shape[1]
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = True

        dec_params = {} | kwargs
        # dec_params['input_size'] = input_shape[1]
        dec_params['input_size'] = 1
        dec_params['hidden_size'] = hidden_size

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        if hidden_size != output_size:
            adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)
            self.adapter = nnx.TimeDistributed(adapter)
        else:
            self.adapter = None

        self._zero_cache: dict[int, torch.Tensor] = {}

    def forward(self, x):
        t, h = self.enc(x)

        z = self._zero(len(x), x)
        t = self.dec(z, h)
        if self.adapter is not None:
            t = self.adapter(t)
        return t

    def _zero(self, batch_size, t):
        # t: tensor used to retrieve dtype and device
        if batch_size not in self._zero_cache:
            # input_size = self.input_shape[1]
            input_size = 1
            output_seqlen = self.output_shape[0]
            zero = torch.zeros((batch_size, output_seqlen, input_size), dtype=t.dtype, device=t.device, requires_grad=False)
            self._zero_cache[batch_size] = zero
        return self._zero_cache[batch_size]
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqV2
# ---------------------------------------------------------------------------
# fit(Xt, (yt, yp))
#

class TSSeq2SeqV2(TimeSeriesModel):
    """
    Simple model:
        [X_train] -> encoder -> [hidden_state] -> decoder -> [y_predict]
                             -> [y_encoder]

    but the encoder output [y_encoder] is compared [y_train]
    """

    _tags = {
        "x-use-ypredict": False,
        "y-use-ytrain": True
    }

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm',
                 use_encoder_sequence=False,
                 **kwargs):
        """

        :param input_shape:
        :param output_shape:
        :param hidden_size:
        :param flavour:
        :param use_sequence: if to use just the last value of the encoder
            or the complete sequence
        :param kwargs:
        """
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         hidden_size=hidden_size,
                         use_encoder_sequence=use_encoder_sequence,
                         **kwargs)

        self.hidden_size = hidden_size
        self.use_encoder_sequence = use_encoder_sequence

        if hidden_size is None:
            hidden_size = output_shape[1]

        input_seqlen = input_shape[0]
        output_size = output_shape[1]

        enc_params = {} | kwargs
        enc_params['input_size'] = input_shape[1]
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = True       # return the hidden state
        enc_params['return_sequence'] = use_encoder_sequence in (True, "return")   # return just the last value

        dec_params = {} | kwargs
        dec_params['input_size'] = hidden_size * (input_seqlen if use_encoder_sequence else 1)
        dec_params['hidden_size'] = hidden_size
        dec_params['return_state'] = False

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        if hidden_size != output_size:
            adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)
            self.adapter = nnx.TimeDistributed(adapter)
        else:
            self.adapter = None
    # end

    def forward(self, x):
        batch_size = len(x)
        output_seqlen = self.output_shape[0]

        # e: encoder sequence
        # t: decoder sequence
        e, h = self.enc(x)
        t = e.reshape(batch_size, -1)

        z = time_repeat(t, output_seqlen)
        t = self.dec(z, h)

        t = t if self.adapter is None else self.adapter(t)

        if self.use_encoder_sequence == 'return':
            e = e if self.adapter is None else self.adapter(e)
            return e, t
        else:
            return t
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
        if hidden_size != output_size:
            adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)
            self.adapter = nnx.TimeDistributed(adapter)
        else:
            self.adapter = None

        # n of times 'forward' is called
        self.iteach = 0
    # end

    def forward(self, xytp):
        if isinstance(xytp, (list, tuple)):
            return self._forward_train(xytp)
        else:
            return self._forward_predict(xytp)

    def _forward_train(self, xyp):
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

    def _forward_predict(self, x):
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
