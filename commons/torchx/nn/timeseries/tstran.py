#
# Transformers for Time Series
#
import torch

from stdlib import kwexclude, kwselect
from .ts import TimeSeriesModel
from ... import nn as nnx
from ...utils import expand_dims


# class TSTransformerV1(TimeSeriesModel):
#
#     def __init__(self, input_shape, output_shape,
#                  rnn_flavour='lstm',
#                  attn_flavour='dot',
#                  **kwargs):
#         super().__init__(input_shape, output_shape,
#                          rnn_flavour=rnn_flavour,
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
#         self.enc = nnx.create_rnn(rnn_flavour, **enc_params)
#         self.dec = nnx.create_rnn(rnn_flavour, **dec_params)
#
#         self.attn = nnx.create_attention(attn_flavour, **attn_params)
#     # end
# # end

class TSTransformerV1(TimeSeriesModel):
    def __init__(self, input_shape, output_shape,
                 nhead=1,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=0,
                 dropout=0,
                 **kwargs):
        super().__init__(input_shape, output_shape, nhead=nhead,
                         num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout, **kwargs)
        input_size = input_shape[1]
        output_size = output_shape[1]
        d_model = nhead*input_size
        if dim_feedforward == 0:
            dim_feedforward = d_model
        self.repl = nnx.PositionalReplicate(nhead, input_size)
        self.tran = nnx.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            **kwargs
        )
        self.adapter = nnx.TimeDistributed(nnx.Linear(d_model, output_size))
        pass
    # end

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return self._forward_train(x)
        else:
            return self._forward_predict(x)

    def _forward_train(self, x):
        x_enc, x_dec = x

        x_enc = self.repl(x_enc)
        x_dec = self.repl(x_dec)

        y_tran = self.tran(x_enc, x_dec)
        # y_enc = self.tran.encoder(x_enc)
        # y_dec = self.tran.decoder(x_dec, y_enc)

        yp = self.adapter(y_tran)
        return yp
    # end

    def _forward_predict(self, x):
        output_seqlen, output_size = self.output_shape

        x_enc = x                           # [N, Lin, Hin]
        x_dec = x[:, -1:, -output_size:]    # [N, 1,  Hout]

        x_enc = self.repl(x_enc)
        x_dec = self.repl(x_dec)

        y_enc = self.tran.encoder(x_enc)

        ylist = []
        for i in range(output_seqlen):
            x_dec = self.repl(x_dec)
            y_pred = self.tran.decoder(x_dec, y_enc)
            y_pred = self.adapter(y_pred)

            ylist.append(y_pred)
            x_dec = y_pred
        # end
        return torch.cat(ylist, dim=1)
    # end
# end
