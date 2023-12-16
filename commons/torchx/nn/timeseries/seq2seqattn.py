import torch
from .ts import TimeSeriesModel
from ... import nn as nnx
from stdlib import kwexclude, kwselect


#
# Native RNN returned hidden state:
#   RNN:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   GRU:    [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#   LSTM:   [D*num_layers, N, Hout]     [D*num_layers, N, L, Hout]
#           [D*num_layers, N, Hcell]    [D*num_layers, N, L, Hcell]
#
#   N: batch size
#   L: sequence length
#   D: 2 if bidi else 1
#   Hin:    input size
#   Hcell:  cell size
#   Hout:   proj_size if proj_size > 0 else hidden size

# ---------------------------------------------------------------------------
# TSSeq2SeqAttnV1
# ---------------------------------------------------------------------------
# Attention as input

def last_hidden_states(h_all):
    # [D*num_layers, N, L, Hout]
    if isinstance(h_all, tuple):
        return h_all[0][:, :, -1, :], h_all[1][:, :, -1, :]
    else:   # h_n
        return h_all[:, :, -1, :]


def last_hidden_state(h_all):
    # [D*num_layers, N, Hout]
    if isinstance(h_all, tuple):
        h_all = h_all[0]
    if len(h_all.shape) == 3:
        # [D*num_layers, N, Hout]
        return h_all
    else:
        # [D*num_layers, N, L, Hout]
        return h_all[-1, :, -1:, :]


def last_layer_states(h_all):
    # [D*num_layers, N, L, Hout]
    if isinstance(h_all, tuple):
        return h_all[0][-1, :, :, :]
    else:
        return h_all[-1, :, :, :]


class TSSeq2SeqAttnV1(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 rnn_flavour='lstm',
                 attn_flavour='dot',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         rnn_flavour=rnn_flavour,
                         attn_flavour=attn_flavour,
                         **kwargs)

        hidden_size = output_shape[1]
        output_size = output_shape[1]

        rnnargs = kwexclude(kwargs, 'attn')

        enc_params = {} | rnnargs
        enc_params['input_size'] = input_shape[1]
        enc_params['hidden_size'] = hidden_size
        enc_params['return_state'] = 'all'  # return all states

        dec_params = {} | rnnargs
        dec_params['input_size'] = hidden_size + hidden_size  # [x, attn]
        dec_params['hidden_size'] = hidden_size
        dec_params['return_state'] = True  # return last state states

        self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
        self.dec = nnx.create_rnn(rnn_flavour, **dec_params)

        attn_params = kwselect(kwargs, 'attn')
        self.attn = nnx.create_attention(attn_flavour, **attn_params)
    # end

    def forward(self, x):
        output_seqlen = self.output_shape[0]

        # [D*num_layers, N,    Hout]    last hidden state
        # [D*num_layers, N, L, Hout]    all hidden states

        e, h_all = self.enc(x)

        # hidden state to pass to the decoder
        h = last_hidden_states(h_all)
        # hidden states (K,V) to pass to the attention
        kv = last_layer_states(h_all)
        # hidden state to pass as Q

        q = last_hidden_state(h_all)
        x = e[:, -1:, :]

        y_list = []
        for i in range(output_seqlen):
            attn = self.attn(q, kv, kv)
            xp = torch.cat([x, attn], dim=-1)

            y, h = self.dec(xp, h)
            y_list.append(y)
            q = last_hidden_state(h_all)

        y_out = torch.cat(y_list, dim=1)
        return y_out
    # end


