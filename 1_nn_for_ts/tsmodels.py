#
# seq2seq + teacher forcing/scheduler sampling
# https://medium.com/@maxbrenner-ai/implementing-seq2seq-models-for-efficient-time-series-forecasting-88dba1d66187
#
# Time Series Cross-validation — a walk forward approach in python
# https://medium.com/eatpredlove/time-series-cross-validation-a-walk-forward-approach-in-python-8534dd1db51a
#
# model based on Linear layer
# 2023 - Are Transformers Effective for Time Series Forecasting
#

# Some extensions on 'fit(X, y)'
#
#   TSLinear        fit(Xt, yp)
#   TSRNNLinear     fit(Xt, yp)
#   TSCNNLinear     fit(Xt, yp)
#   TSSeq2SeqV1     fit(Xt, yp)
#   TSSeq2SeqV2     fit(Xt, (yt, yp))
#   TSSeq2SeqV3     fit((Xt, yp), (yt, yp))
#


import torch

import torchx.nn as nnx
import torchx
from stdlib import kwparams, kwexclude, kwval
from is_instance import is_instance
from math import exp
from random import random


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

def create_model(name: str, input_shape, output_shape, **kwargs):

    if name == 'linear':
        return TSLinear(input_shape, output_shape, **kwargs)
    if name == 'rnnlin':
        return TSRNNLinear(input_shape, output_shape, **kwargs)
    if name == 'cnnlin':
        return TSCNNLinear(input_shape, output_shape, **kwargs)
    if name == 'seq2seq1':
        return TSSeq2SeqV1(input_shape, output_shape, **kwargs)
    if name == 'seq2seq2':
        return TSSeq2SeqV2(input_shape, output_shape, **kwargs)
    if name == 'seq2seq3':
        return TSSeq2SeqV3(input_shape, output_shape, **kwargs)
    if name == 'seq2seq_attn1':
        return TSSeq2SeqAttnV1(input_shape, output_shape, **kwargs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_shape(s):
    # return isinstance(s, tuple) and (len(s) == 2)
    return is_instance(s, tuple[int, int])


# ---------------------------------------------------------------------------
# TimeSeriesModel
# ---------------------------------------------------------------------------
# Tags
#   x-use-ypredict == True      fit((Xt, yp), ...)
#   y-use-ytrain   == True      fit(..., (yt, yp))
# Parameters
#   target_first   == True      X = [yt, xt]
#   target_first   == False     X = [xt, yt]
#

class TimeSeriesModel(nnx.Module):
    _tags = {
        "x-use-ypredict": False,
        "y-use-ytrain": False
    }

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        assert is_shape(input_shape), "input_shape"
        assert is_shape(output_shape), "output_shape"

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kwargs = kwargs
# end


# ---------------------------------------------------------------------------
# TSLinearModel
# ---------------------------------------------------------------------------

class TSLinear(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 activation='relu', **kwargs):
        """
        Time series Linear model.

        :param input_shape: sequence_length x n_input_features
        :param output_shape: sequence_length x n_target_features
        :param hidden_size: hidden size
        :param activation: activation function to use
        :param kwargs: parameters to use for the activation function.
                       They must be 'activation__<parameter_name>'
        """
        super().__init__(input_shape, output_shape, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_params = kwparams(kwargs, 'activation')

        if hidden_size is not None:
            self.encoder = nnx.Linear(in_features=input_shape, out_features=hidden_size)
            self.relu = torchx.activation_function(self.activation, self.activation_params)
            self.decoder = nnx.Linear(in_features=hidden_size, out_features=output_shape)
        else:
            self.encoder = nnx.Linear(in_features=input_shape, out_features=output_shape)
            self.decoder = None
            self.relu = None
        # end

    def forward(self, x):
        if self.hidden_size is None:
            t = self.encoder(x)
        else:
            t = self.encoder(x)
            t = self.relu(t)
            t = self.decoder(t)
        return t
# end


# ---------------------------------------------------------------------------
# TSRecurrentLinear
# ---------------------------------------------------------------------------

class TSRNNLinear(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm', activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         hidden_size=hidden_size,
                         **kwargs)
        self.hidden_size = hidden_size
        self.flavour = flavour

        if hidden_size is None:
            hidden_size = input_shape[1]

        activation_params = kwparams(kwargs, 'activation')

        rnn_params = kwexclude(kwargs, 'activation')
        rnn_params['input_size'] = input_shape[1]
        rnn_params['hidden_size'] = hidden_size

        input_seqlen = input_shape[0]

        self.rnn = nnx.create_rnn(flavour, **rnn_params)
        self.relu = torchx.activation_function(activation, activation_params)
        self.linear = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = self.rnn(x)
        t = self.relu(t) if self.relu is not None else t
        t = self.linear(t)
        return t
# end


# ---------------------------------------------------------------------------
# TSConvolutionalLinear
# ---------------------------------------------------------------------------

class TSCNNLinear(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='cnn', activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         hidden_size=hidden_size,
                         **kwargs)

        self.hidden_size = hidden_size
        self.flavour = flavour

        if hidden_size is None:
            hidden_size = input_shape[1]

        activation_params = kwparams(kwargs, 'activation')
        cnn_params = kwexclude(kwargs, 'activation')
        # Force the tensor layout equals to the RNN layers
        cnn_params['in_channels'] = input_shape[1]
        cnn_params['out_channels'] = hidden_size
        cnn_params['channels_last'] = True

        input_seqlen = input_shape[0]

        self.cnn = nnx.create_cnn(flavour, **cnn_params)
        self.relu = torchx.activation_function(activation, activation_params)
        self.linear = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = self.cnn(x)
        t = self.relu(t) if self.relu is not None else t
        t = self.linear(t)
        return t
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqV1
# ---------------------------------------------------------------------------

class TSSeq2SeqV1(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         hidden_size=hidden_size,
                         **kwargs)

        self.hidden_size = hidden_size
        self.flavour = flavour

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
        self.flavour = flavour
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

        z = torchx.time_repeat(t, output_seqlen)
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
        self.flavour = flavour
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
        if isinstance(xytp, (tuple, list)):
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


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV1
# ---------------------------------------------------------------------------

class TSSeq2SeqAttnV1(TimeSeriesModel):
    pass


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
