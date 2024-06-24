
__all__ = [
    "TSSeq2Seq"
]

import torch
from .ts import TimeSeriesModel
from .tsutils import TimeLinear, ZeroCache, apply_if
from ... import nn as nnx


# ---------------------------------------------------------------------------
# TSSeq2Seq
# ---------------------------------------------------------------------------
# 'decoder_mode': what to pass as input to the decoder
#   None, "zero"    the zero vector
#   "last"          the last encoder output, replicated
#   "sequence"      all encoder output, flatten and replicated
#   "adapt"         all encoder output, adapted as decoder input
#   "recursive"     the last encoder output used as first decoder input
#                   the decoder output used as next decoder input
#   "hs-recursive"  as "recursive" but with the encoder hidden state
#                   concatenated to the input, and transformed

class TSSeq2Seq(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 decoder_mode=None,
                 flavour='lstm',
                 nonlinearity='tanh',
                 **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         nonlinearity=nonlinearity,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         **kwargs)

        input_seqlen, input_size = input_shape
        output_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.decoder_mode = decoder_mode

        enc_params = {} | kwargs
        enc_params['input_size'] = feature_size
        enc_params['hidden_size'] = hidden_size
        enc_params['nonlinearity'] = nonlinearity
        enc_params['return_state'] = True
        enc_params['return_sequence'] = decoder_mode in ["sequence", "adapt"]

        dec_params = {} | kwargs
        dec_params['input_size'] = feature_size
        dec_params['hidden_size'] = hidden_size
        enc_params['nonlinearity'] = nonlinearity
        dec_params['return_state'] = decoder_mode in ["recursive", "hs-recursive"]

        self.enc = nnx.create_rnn(flavour, **enc_params)
        self.dec = nnx.create_rnn(flavour, **dec_params)
        self.enc_input_adapter = None
        self.dec_input_adapter = None
        self.dec_output_adapter = None

        if input_size != feature_size:
            self.enc_input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)
        if hidden_size != output_size:
            self.dec_output_adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)

        if decoder_mode in [None, "", "zero", "none"]:
            self.dec_input_adapter = ZeroCache((output_seqlen, feature_size))
        elif decoder_mode == "last":
            self.dec_input_adapter = TimeLinear(in_features=(1, hidden_size), out_features=(output_seqlen, feature_size))
        elif decoder_mode == "sequence":
            self.dec_input_adapter = TimeLinear(in_features=(input_seqlen, hidden_size), out_features=(output_seqlen, feature_size))
        elif decoder_mode == "adapt":
            self.dec_input_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=(output_seqlen, feature_size))
        elif decoder_mode == "recursive":
            self.dec_input_adapter = nnx.Linear(in_features=hidden_size, out_features=feature_size)
        elif decoder_mode == "hs-recursive":
            self.dec_input_adapter = nnx.Linear(in_features=(hidden_size + hidden_size), out_features=feature_size)
        else:
            raise ValueError(f"Unsupported 'decoder_mode': {decoder_mode}")
    # end

    def forward(self, x):
        if self.decoder_mode == "recursive":
            return self._recursive_forward(x, False)
        elif self.decoder_mode == "hs-recursive":
            return self._recursive_forward(x, True)
        else:
            return self._single_forward(x)

    def _single_forward(self, x):
        t = apply_if(x, self.enc_input_adapter)

        e, h = self.enc(t)
        d = apply_if(e, self.dec_input_adapter)
        t = self.dec(d, h)

        y = apply_if(t, self.dec_output_adapter)
        return y

    def _recursive_forward(self, x, use_hidden_state):
        output_seqlen = self.output_shape[0]
        t = apply_if(x, self.enc_input_adapter)

        e, h = self.enc(t)

        # expand 'e' adding the 'seqlen' dimension, equals to 1
        t = e[:, None, :]

        # LSTM returns 2 hidden states. It will be used just the first one
        # if the RNN is composed by multiple layers, it is used the hidden state of the last one
        # expand 'hidden_state' adding the 'seqlen' dimension, equals to 1
        hidden_state = h[0] if isinstance(h, tuple) else h
        hidden_state = hidden_state[-1]
        hidden_state = hidden_state[:, None, :]

        ylist = []
        for i in range(output_seqlen):

            if use_hidden_state:
                # concatenate t and hidden state along the 'data' axis
                t = torch.concat([t, hidden_state], dim=2)

            x = self.dec_input_adapter(t)
            y, h = self.dec(x, h)
            ylist.append(y)
            t = y
        t = torch.cat(ylist, dim=1)
        y = apply_if(t, self.dec_output_adapter)
        return y
# end


# ---------------------------------------------------------------------------
# TSSeq2SeqV3
# ---------------------------------------------------------------------------
# fit((Xt, yp), (yt, yp))
#

# class TSSeq2SeqV3(TimeSeriesModel):
#     """
#     Simple model:
#         [X_train] -> encoder -> [hidden_state] -> decoder -> [y_predict]
#                              -> [y_encoder]
#
#     but the encoder output [y_encoder] is compared [y_train] AND it is used
#     the mechanism 'teacher forcing'
#     """
#
#     _tags = {
#         "x-use-ypredict": True,
#         "y-use-ytrain": True
#     }
#
#     def __init__(self, input_shape, output_shape,
#                  feature_size=None,
#                  hidden_size=None,
#                  flavour='lstm',
#                  target_first=True,
#                  teacher_forcing=0.01,
#                  **kwargs):
#         """
#
#         :param input_shape:
#         :param output_shape:
#         :param hidden_size:
#         :param flavour:
#         :param target_first: the the target is present in the first (True) or last (False) columns
#             of the tensor X used for the training/prediction
#         :param teacher_forcing: decay exponent used for the 'teacher forcing' mechanism.
#         :param kwargs:
#         """
#         super().__init__(input_shape, output_shape,
#                          flavour=flavour,
#                          feature_size=feature_size,
#                          hidden_size=hidden_size,
#                          target_first=target_first,
#                          teacher_forcing=teacher_forcing,
#                          **kwargs)
#
#         input_seqlen, input_size = input_shape
#         output_seqlen, output_size = output_shape
#
#         if feature_size is None:
#             feature_size = input_size
#         if hidden_size is None:
#             hidden_size = feature_size
#
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.target_first = target_first
#         self.teacher_forcing = teacher_forcing
#
#         enc_params = {} | kwargs
#         enc_params['input_size'] = feature_size
#         enc_params['hidden_size'] = hidden_size
#         enc_params['return_state'] = True       # return the hidden state
#         enc_params['return_sequence'] = True    # return all encoder sequence
#
#         dec_params = {} | kwargs
#         dec_params['input_size'] = feature_size
#         dec_params['hidden_size'] = hidden_size
#         dec_params['return_state'] = True
#
#         self.enc = nnx.create_rnn(flavour, **enc_params)
#         self.dec = nnx.create_rnn(flavour, **dec_params)
#         self.enc_input_adapter = None
#         self.dec_input_adapter = None
#         self.dec_output_adapter = None
#
#         if input_size != feature_size:
#             self.enc_input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)
#         self.dec_input_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=(output_seqlen, feature_size))
#         if feature_size != output_size:
#             self.dec_output_adapter = nnx.Linear(in_features=hidden_size, out_features=output_size)
#
#         # n of times 'forward' is called
#         self.iteach = 0
#     # end
#
#     def forward(self, xytp):
#         # if isinstance(xytp, (list, tuple)):
#         #     return self._train_forward(xytp)
#         # else:
#         #     return self._predict_forward(xytp)
#         return self._forward(xytp)
#
#     def _forward(self, x):
#
#         t = apply_if(x, self.enc_input_adapter)
#
#
#         pass
#
#
#     def _train_forward(self, xyp):
#         # xyp = (X, y_pred)
#         x, yp = xyp
#         # n of features in y
#         output_seqlen, my = self.output_shape
#         self.iteach += 1
#
#         t = apply_if(x, self.enc_input_adapter)
#         e, h = self.enc(t)
#
#         # yprev
#         if self.target_first:
#             yprev = x[:, -1:, :my]
#         else:
#             yprev = x[:, -1:, -my:]
#
#         ylist = []
#         for i in range(output_seqlen):
#             if i > 0 and self.teacher_forcing > 0:
#                 prob = exp(-self.iteach * self.teacher_forcing)
#                 if random() < prob:
#                     # yprev[:, :, :] = yp[:, i-1:i, :]
#                     yprev.data = torch.clone(yp[:, i-1:i, :]).data
#                 else:
#                     pass
#
#             yc, h = self.dec(yprev, h)
#             yc = yc if self.adapter is None else self.adapter(yc)
#             ylist.append(yc)
#             yprev = yc
#
#         t = torch.cat(ylist, dim=1)
#         return e, t
#
#     def _predict_forward(self, x):
#         e, h = self.enc(x)
#         e = e if self.adapter is None else self.adapter(e)
#         output_seqlen, my = self.output_shape
#
#         # yprev
#         if self.target_first:
#             yprev = e[:, -1:, :my]
#         else:
#             yprev = e[:, -1:, -my:]
#
#         ylist = []
#         for i in range(output_seqlen):
#             yc, h = self.dec(yprev, h)
#             yc = yc if self.adapter is None else self.adapter(yc)
#             ylist.append(yc)
#             yprev = yc
#
#         t = torch.cat(ylist, dim=1)
#         return t
# # end
