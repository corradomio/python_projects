#
# Sequence to sequence for Time Series
#
import torch

from stdlib import kwexclude, kwselect
from .ts import TimeSeriesModel
from ... import nn as nnx
from ...utils import expand_dims


#
# Native RNN returned hidden state:
#           last cell                   all cells (collected)
#   RNN:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   GRU:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   LSTM:   [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#           [D*num_layers, N, Hcell]    [D*num_layers, N, L, Hcell]
#

# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV1
# ---------------------------------------------------------------------------
# encoder hidden state as (K, V)
# attention as decoder input

class TSSeq2SeqAttnV1(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 rnn_flavour='lstm',
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         rnn_flavour=rnn_flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        input_size = input_shape[1]
        output_size = output_shape[1]

        rnnargs = kwexclude(kwargs, 'attn')
        attn_params = kwselect(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = input_size
        enc_params['hidden_size'] = output_size
        enc_params['return_state'] = 'all'                      # return all states

        dec_params = {} | rnnargs
        dec_params['input_size'] = output_size + output_size    # combine [attention|y]
        dec_params['hidden_size'] = output_size
        dec_params['return_state'] = True                      # must return the state to use in the next cell

        self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
        self.dec = nnx.create_rnn(rnn_flavour, **dec_params)

        self.attn = nnx.create_attention(attn_flavour, **attn_params)
    # end

    def forward(self, x):
        output_len, output_size = self.output_shape

        # e: encoder sequence [N, Lin, Hin]
        # h: hidden state with size [D*num_layers, N, Lin, Hin]
        #    a 2-tuple for LSTM
        # yout: decoder sequence
        e, h = self.enc(x)
        if isinstance(h, tuple):
            hall = h[0]
            # hidden state to pass to the decoder
            hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
        else:
            hall = h
            # hidden state to pass to the decoder
            hn = h[0][:, :, -1, :]

        # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
        # but what to use as K,V,q?
        K = hall[-1]
        V = K
        q = hall[-1, :, -1:, :]

        # extract the last prediction from the encoder
        y = e[:, -1:, :]

        yout = []
        for i in range(output_len):
            # call the attention mechanism
            a = self.attn(q, K, V)

            # 'x decoder': concatenate 'a' with 'y'
            xd = torch.cat([a,y], dim=2)

            yi, hi = self.dec(xd, hn)
            yout.append(yi)

            hn = hi
            q = expand_dims(hi[0][-1], 1)
            y = yi
        # end

        yout = torch.cat(yout, dim=1)
        return yout
    # end
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV3
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

class TSSeq2SeqAttnV3(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 rnn_flavour='lstm',
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         rnn_flavour=rnn_flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        input_size = input_shape[1]
        output_size = output_shape[1]

        rnnargs = kwexclude(kwargs, 'attn')
        attn_params = kwselect(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = input_size
        enc_params['hidden_size'] = output_size
        enc_params['return_state'] = 'all'                      # return all states

        dec_params = {} | rnnargs
        dec_params['input_size'] = output_size + output_size    # combine [attention|y]
        dec_params['hidden_size'] = output_size
        dec_params['return_state'] = True                      # must return the state to use in the next cell

        self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
        self.dec = nnx.create_rnn(rnn_flavour, **dec_params)

        self.attn = nnx.create_attention(attn_flavour, **attn_params)
    # end

    def forward(self, x):
        output_len, output_size = self.output_shape

        # e: encoder sequence [N, Lin, Hin]
        # h: hidden state with size [D*num_layers, N, Lin, Hin]
        #    a 2-tuple for LSTM
        # yout: decoder sequence
        e, h = self.enc(x)
        if isinstance(h, tuple):
            hall = h[0]
            # hidden state to pass to the decoder
            hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
        else:
            hall = h
            # hidden state to pass to the decoder
            hn = h[0][:, :, -1, :]

        # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
        # but what to use as K,V,q?
        K = hall[-1]
        V = e
        q = hall[-1, :, -1:, :]

        # extract the last prediction from the encoder
        y = e[:, -1:, :]

        yout = []
        for i in range(output_len):
            # call the attention mechanism
            a = self.attn(q, K, V)

            # 'x decoder': concatenate 'a' with 'y'
            xd = torch.cat([a,y], dim=2)

            yi, hi = self.dec(xd, hn)
            yout.append(yi)

            hn = hi
            q = expand_dims(hi[0][-1], 1)
            y = yi
        # end

        yout = torch.cat(yout, dim=1)
        return yout
    # end
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV2
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

class TSSeq2SeqAttnV2(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 rnn_flavour='lstm',
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         rnn_flavour=rnn_flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        input_size = input_shape[1]
        output_size = output_shape[1]

        rnnargs = kwexclude(kwargs, 'attn')
        attn_params = kwselect(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = input_size
        enc_params['hidden_size'] = output_size
        enc_params['return_state'] = 'all'  # return all states

        dec_params = {} | rnnargs
        dec_params['input_size'] = output_size + output_size  # combine [attention|y]
        dec_params['hidden_size'] = output_size
        dec_params['return_state'] = True  # must return the state to use in the next cell

        self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
        self.dec = nnx.create_rnn(rnn_flavour, **dec_params)

        self.attn = nnx.create_attention(attn_flavour, **attn_params)

    # end

    def forward(self, x):
        output_len, output_size = self.output_shape

        # e: encoder sequence [N, Lin, Hin]
        # h: hidden state with size [D*num_layers, N, Lin, Hin]
        #    a 2-tuple for LSTM
        # yout: decoder sequence
        e, h = self.enc(x)
        if isinstance(h, tuple):
            hall = h[0]
            # hidden state to pass to the decoder
            hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
        else:
            hall = h
            # hidden state to pass to the decoder
            hn = h[0][:, :, -1, :]

        # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
        # but what to use as K,V,q?
        K = hall[-1]
        V = K
        q = hall[-1, :, -1:, :]

        # extract the last prediction from the encoder
        y = e[:, -1:, :]

        yout = []
        for i in range(output_len):
            # call the attention mechanism
            a = self.attn(q, K, V)

            yi, hi = self.dec(y, a)
            yout.append(yi)

            hn = hi
            q = expand_dims(hi[0][-1], 1)
            y = yi
        # end

        yout = torch.cat(yout, dim=1)
        return yout
    # end
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV2
# ---------------------------------------------------------------------------
# encoder hidden state K
# encoder output as V
# attention as decoder input

class TSSeq2SeqAttnV4(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 rnn_flavour='lstm',
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         rnn_flavour=rnn_flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        input_size = input_shape[1]
        output_size = output_shape[1]

        rnnargs = kwexclude(kwargs, 'attn')
        attn_params = kwselect(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = input_size
        enc_params['hidden_size'] = output_size
        enc_params['return_state'] = 'all'  # return all states

        dec_params = {} | rnnargs
        dec_params['input_size'] = output_size + output_size  # combine [attention|y]
        dec_params['hidden_size'] = output_size
        dec_params['return_state'] = True  # must return the state to use in the next cell

        self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
        self.dec = nnx.create_rnn(rnn_flavour, **dec_params)

        self.attn = nnx.create_attention(attn_flavour, **attn_params)

    # end

    def forward(self, x):
        output_len, output_size = self.output_shape

        # e: encoder sequence [N, Lin, Hin]
        # h: hidden state with size [D*num_layers, N, Lin, Hin]
        #    a 2-tuple for LSTM
        # yout: decoder sequence
        e, h = self.enc(x)
        if isinstance(h, tuple):
            hall = h[0]
            # hidden state to pass to the decoder
            hn = (h[0][:, :, -1, :], h[1][:, :, -1, :])
        else:
            hall = h
            # hidden state to pass to the decoder
            hn = h[0][:, :, -1, :]

        # hall [D*num_layers, N, L, Hout] is used as K and V for the attention module
        # but what to use as K,V,q?
        K = hall[-1]
        V = K
        q = hall[-1, :, -1:, :]

        # extract the last prediction from the encoder
        y = e[:, -1:, :]

        yout = []
        for i in range(output_len):
            # call the attention mechanism
            a = self.attn(q, K, V)

            yi, hi = self.dec(y, a)
            yout.append(yi)

            hn = hi
            q = expand_dims(hi[0][-1], 1)
            y = yi
        # end

        yout = torch.cat(yout, dim=1)
        return yout
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
