#
# Sequence to sequence for Time Series
#

__all__ = [
    "TSSeq2SeqAttn",
]

import torch

from stdlib import kwexclude, kwparams
from .ts import TimeSeriesModel
from .tsutils import apply_if
from ... import nn as nnx

#
# Native RNN returned hidden state:
#           last cell                   all cells (collected)
#   RNN:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   GRU:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   LSTM:   [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#           [D*num_layers, N, Hcell]    [D*num_layers, N, L, Hcell]
#

def _split_hidden_state(h):
    if isinstance(h, tuple):
        hall = h[0]
        # hidden state to pass to the decoder: le last state
        hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
    else:
        hall = h
        # hidden state to pass to the decoder
        hn = h[0][:, :, -1, :]
    return hall, hn


def _compose_hidden_state(h, a):
    # 32,1,8 -> 1,32,8
    batch_size, seq_len, data_size = a.shape
    a = a.reshape((seq_len, batch_size, data_size))
    if isinstance(h, tuple):
        return a, h[1]
    else:
        return a


def _last_hidden_state(h):
    if isinstance(h, tuple):
        hs = h[0][-1]
    else:
        hs = h[-1]
    return hs[:, None, :]


# ---------------------------------------------------------------------------
# TSSeq2SeqAttn
# ---------------------------------------------------------------------------
# encoder hidden state as (K, V)
# Sequence 2 sequence with attention mechanism
#
#   'attn__*' are used as parameters to pass to Attention constructor
#
# attn_input: data to pass to attention input (Key, Value)
#   None, False     hidden state as Key and Value
#   True            hidden state as Key, encoder output as Value
#
# attn_output: how to use the attention output
#   None, False     output concatenated to the decoder input
#   True            output used as decoder hidden state
#


class TSSeq2SeqAttn(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 flavour='lstm',
                 attn_input=False,
                 attn_output=False,
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         flavour=flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        input_seqlen, input_size = input_shape
        output_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.attn_input = attn_input
        self.att_output = attn_output

        rnnargs = kwexclude(kwargs, 'attn')
        attn_params = kwparams(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = feature_size
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = 'all'                  # return all states
        enc_params['return_sequence'] = True                # if True, Attention requires fullr encoder's output
                                                            # and full hidden state
        dec_params = {} | rnnargs
        # combine [attention|y] or y only
        dec_params['input_size'] = feature_size
        dec_params['hidden_size'] = hidden_size
        dec_params['return_state'] = True                       # must return the state to use in the next cell

        # create the attention mechanism based on the flavour
        self.attn = nnx.create_attention(attn_flavour, **attn_params)

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        self.enc_input_adapter = None
        self.dec_input_adapter = None
        self.dec_output_adapter = None

        if input_size != feature_size:
            self.enc_input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)
        if hidden_size != output_size:
            self.dec_output_adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)

        if not attn_output:
            # [False] concatenate the attention output with the decoder input
            self.dec_input_adapter = nnx.Linear(in_features=(hidden_size+hidden_size), out_features=feature_size)
        elif hidden_size != feature_size:
            # [True] the attention is used as hidden state
            self.dec_input_adapter = nnx.Linear(in_features=hidden_size, out_features=feature_size)
        else:
            self.dec_input_adapter = None
    # end

    def forward(self, x):
        output_len, output_size = self.output_shape

        t = apply_if(x, self.enc_input_adapter)

        e, h = self.enc(t)
        # e: encoder sequence [N, Lin, Hin]
        # h: hidden state with size [D*num_layers, N, Lin, Hin]
        #    a 2-tuple for LSTM
        # It is used only the hidden state of the last layer

        hall, hn = _split_hidden_state(h)

        # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
        # but what to use as K,V,q?
        if self.attn_input:
            K = hall[-1]    # hidden state of the last layer
            V = e           # encoder output
        else:
            K = hall[-1]    # hidden state of the last layer
            V = K           # hidden state of the last layer

        # query: last hidden state of the last layer
        q = hall[-1, :, -1:, :]

        # last prediction from the encoder's output
        y = e[:, -1:, :]

        yout = []
        for i in range(output_len):
            # call the attention mechanism
            a = self.attn(q, K, V)

            # use the attention
            if not self.att_output:
                # [False] attention as decoder input
                xd = torch.concat([y, a], dim=2)
            else:
                # [True] attention as hidden state
                hn = _compose_hidden_state(hn, a)
                xd = y

            xd = apply_if(xd, self.dec_input_adapter)

            yi, hi = self.dec(xd, hn)
            yout.append(yi)

            hn = hi
            q = _last_hidden_state(hn)
            y = yi
        # end

        yout = torch.cat(yout, dim=1)

        y = apply_if(yout, self.dec_output_adapter)

        return y
    # end
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV1
# ---------------------------------------------------------------------------
# encoder hidden state as (K, V)
# attention as decoder input

# class TSSeq2SeqAttnV1(TimeSeriesModel):
#
#     def __init__(self, input_shape, output_shape,
#                  flavour='lstm',
#                  attn_flavour='dot',
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          flavour=flavour,
#                          attn_flavour=attn_flavour,
#                          **kwargs)
#
#         input_size = input_shape[1]
#         output_size = output_shape[1]
#
#         rnnargs = kwexclude(kwargs, 'attn')
#         attn_params = kwselect(kwargs, 'attn')
#
#         enc_params = {} | rnnargs
#         enc_params['input_size'] = input_size
#         enc_params['hidden_size'] = output_size
#         enc_params['return_state'] = 'all'                      # return all states
#
#         dec_params = {} | rnnargs
#         dec_params['input_size'] = output_size + output_size    # combine [attention|y]
#         dec_params['hidden_size'] = output_size
#         dec_params['return_state'] = True                      # must return the state to use in the next cell
#
#         self.enc = nnx.create_rnn(flavour, **enc_params)
#         self.dec = nnx.create_rnn(flavour, **dec_params)
#
#         self.attn = nnx.create_attention(attn_flavour, **attn_params)
#     # end
#
#     def forward(self, x):
#         output_len, output_size = self.output_shape
#
#         # e: encoder sequence [N, Lin, Hin]
#         # h: hidden state with size [D*num_layers, N, Lin, Hin]
#         #    a 2-tuple for LSTM
#         # yout: decoder sequence
#         e, h = self.enc(x)
#         if isinstance(h, tuple):
#             hall = h[0]
#             # hidden state to pass to the decoder
#             hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
#         else:
#             hall = h
#             # hidden state to pass to the decoder
#             hn = h[0][:, :, -1, :]
#
#         # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
#         # but what to use as K,V,q?
#         K = hall[-1]
#         V = K
#         q = hall[-1, :, -1:, :]
#
#         # extract the last prediction from the encoder
#         y = e[:, -1:, :]
#
#         yout = []
#         for i in range(output_len):
#             # call the attention mechanism
#             a = self.attn(q, K, V)
#
#             # 'x decoder': concatenate 'a' with 'y'
#             xd = torch.cat([a,y], dim=2)
#
#             yi, hi = self.dec(xd, hn)
#             yout.append(yi)
#
#             hn = hi
#             q = expand_dims(hi[0][-1], 1)
#             y = yi
#         # end
#
#         yout = torch.cat(yout, dim=1)
#         return yout
#     # end
# # end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV2
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

# class TSSeq2SeqAttnV2(TimeSeriesModel):
#
#     def __init__(self, input_shape, output_shape,
#                  flavour='lstm',
#                  attn_flavour='dot',
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          flavour=flavour,
#                          attn_flavour=attn_flavour,
#                          **kwargs)
#
#         input_size = input_shape[1]
#         output_size = output_shape[1]
#
#         rnnargs = kwexclude(kwargs, 'attn')
#         attn_params = kwselect(kwargs, 'attn')
#
#         enc_params = {} | rnnargs
#         enc_params['input_size'] = input_size
#         enc_params['hidden_size'] = output_size
#         enc_params['return_state'] = 'all'                      # return all states
#
#         dec_params = {} | rnnargs
#         dec_params['input_size'] = output_size + output_size    # combine [attention|y]
#         dec_params['hidden_size'] = output_size
#         dec_params['return_state'] = True                      # must return the state to use in the next cell
#
#         self.enc = nnx.create_rnn(flavour, **enc_params)
#         self.dec = nnx.create_rnn(flavour, **dec_params)
#
#         self.attn = nnx.create_attention(attn_flavour, **attn_params)
#     # end
#
#     def forward(self, x):
#         output_len, output_size = self.output_shape
#
#         # e: encoder sequence [N, Lin, Hin]
#         # h: hidden state with size [D*num_layers, N, Lin, Hin]
#         #    a 2-tuple for LSTM
#         # yout: decoder sequence
#         e, h = self.enc(x)
#         if isinstance(h, tuple):
#             hall = h[0]
#             # hidden state to pass to the decoder
#             hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
#         else:
#             hall = h
#             # hidden state to pass to the decoder
#             hn = h[0][:, :, -1, :]
#
#         # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
#         # but what to use as K,V,q?
#         K = hall[-1]
#         V = e
#         q = hall[-1, :, -1:, :]
#
#         # extract the last prediction from the encoder
#         y = e[:, -1:, :]
#
#         yout = []
#         for i in range(output_len):
#             # call the attention mechanism
#             a = self.attn(q, K, V)
#
#             # 'x decoder': concatenate 'a' with 'y'
#             xd = torch.cat([a,y], dim=2)
#
#             yi, hi = self.dec(xd, hn)
#             yout.append(yi)
#
#             hn = hi
#             q = expand_dims(hi[0][-1], 1)
#             y = yi
#         # end
#
#         yout = torch.cat(yout, dim=1)
#         return yout
#     # end
# # end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV3
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

# class TSSeq2SeqAttnV3(TimeSeriesModel):
#
#     def __init__(self, input_shape, output_shape,
#                  flavour='lstm',
#                  attn_flavour='dot',
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          flavour=flavour,
#                          attn_flavour=attn_flavour,
#                          **kwargs)
#
#         input_size = input_shape[1]
#         output_size = output_shape[1]
#
#         rnnargs = kwexclude(kwargs, 'attn')
#         attn_params = kwselect(kwargs, 'attn')
#
#         enc_params = {} | rnnargs
#         enc_params['input_size'] = input_size
#         enc_params['hidden_size'] = output_size
#         enc_params['return_state'] = 'all'  # return all states
#
#         dec_params = {} | rnnargs
#         dec_params['input_size'] = output_size + output_size  # combine [attention|y]
#         dec_params['hidden_size'] = output_size
#         dec_params['return_state'] = True  # must return the state to use in the next cell
#
#         self.enc = nnx.create_rnn(flavour, **enc_params)
#         self.dec = nnx.create_rnn(flavour, **dec_params)
#
#         self.attn = nnx.create_attention(attn_flavour, **attn_params)
#
#     # end
#
#     def forward(self, x):
#         output_len, output_size = self.output_shape
#
#         # e: encoder sequence [N, Lin, Hin]
#         # h: hidden state with size [D*num_layers, N, Lin, Hin]
#         #    a 2-tuple for LSTM
#         # yout: decoder sequence
#         e, h = self.enc(x)
#         if isinstance(h, tuple):
#             hall = h[0]
#             # hidden state to pass to the decoder
#             hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
#         else:
#             hall = h
#             # hidden state to pass to the decoder
#             hn = h[0][:, :, -1, :]
#
#         # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
#         # but what to use as K,V,q?
#         K = hall[-1]
#         V = K
#         q = hall[-1, :, -1:, :]
#
#         # extract the last prediction from the encoder
#         y = e[:, -1:, :]
#
#         yout = []
#         for i in range(output_len):
#             # call the attention mechanism
#             a = self.attn(q, K, V)
#
#             yi, hi = self.dec(y, a)
#             yout.append(yi)
#
#             hn = hi
#             q = expand_dims(hi[0][-1], 1)
#             y = yi
#         # end
#
#         yout = torch.cat(yout, dim=1)
#         return yout
#     # end
# # end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV4
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

# class TSSeq2SeqAttnV4(TimeSeriesModel):
#
#     def __init__(self, input_shape, output_shape,
#                  flavour='lstm',
#                  attn_flavour='dot',
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          flavour=flavour,
#                          attn_flavour=attn_flavour,
#                          **kwargs)
#
#         input_size = input_shape[1]
#         output_size = output_shape[1]
#
#         rnnargs = kwexclude(kwargs, 'attn')
#         attn_params = kwselect(kwargs, 'attn')
#
#         enc_params = {} | rnnargs
#         enc_params['input_size'] = input_size
#         enc_params['hidden_size'] = output_size
#         enc_params['return_state'] = 'all'  # return all states
#
#         dec_params = {} | rnnargs
#         dec_params['input_size'] = output_size + output_size  # combine [attention|y]
#         dec_params['hidden_size'] = output_size
#         dec_params['return_state'] = True  # must return the state to use in the next cell
#
#         self.enc = nnx.create_rnn(flavour, **enc_params)
#         self.dec = nnx.create_rnn(flavour, **dec_params)
#
#         self.attn = nnx.create_attention(attn_flavour, **attn_params)
#
#     # end
#
#     def forward(self, x):
#         output_len, output_size = self.output_shape
#
#         # e: encoder sequence [N, Lin, Hin]
#         # h: hidden state with size [D*num_layers, N, Lin, Hin]
#         #    a 2-tuple for LSTM
#         # yout: decoder sequence
#         e, h = self.enc(x)
#         if isinstance(h, tuple):
#             hall = h[0]
#             # hidden state to pass to the decoder
#             hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
#         else:
#             hall = h
#             # hidden state to pass to the decoder
#             hn = h[0][:, :, -1, :]
#
#         # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
#         # but what to use as K,V,q?
#         K = hall[-1]
#         V = K
#         q = hall[-1, :, -1:, :]
#
#         # extract the last prediction from the encoder
#         y = e[:, -1:, :]
#
#         yout = []
#         for i in range(output_len):
#             # call the attention mechanism
#             a = self.attn(q, K, V)
#
#             yi, hi = self.dec(y, a)
#             yout.append(yi)
#
#             hn = hi
#             q = expand_dims(hi[0][-1], 1)
#             y = yi
#         # end
#
#         yout = torch.cat(yout, dim=1)
#         return yout
#     # end
# # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
