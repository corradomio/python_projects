import torch
import torch.nn as nn

from ..activation import activation_function

# ---------------------------------------------------------------------------
# RNN/GRU/LSTM
# ---------------------------------------------------------------------------
# It combines in a module a RNN layer connected to a Linear layer,
# batch_first = True for default


#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two LSTMs together to form a `stacked LSTM`,
#             with the second LSTM taking in outputs of the first LSTM and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             LSTM layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
#         proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
#
class LSTM(nn.LSTM):

    def __init__(self, *,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 output_size=1,
                 steps=1,
                 activation=None,
                 activation_params=None,
                 batch_first=True,
                 **kwargs):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)
        self.steps = steps
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f * hidden_size * steps, out_features=output_size)
        else:
            self.V = None
            self.output_size = f * hidden_size * steps

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]

        if L not in self.hidden:
            # D = self.D
            # N = self.hidden_size
            # hidden_state = torch.zeros(D, L, N, dtype=input.dtype)
            # cell_state = torch.zeros(D, L, N, dtype=input.dtype)
            # self.hidden[L] = (hidden_state, cell_state)
            self.hidden[L] = None

        hidden = self.hidden[L]
        t, h = super().forward(input, hidden)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        output = self.V(t) if self.V else t
        return output
# end


#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two GRUs together to form a `stacked GRU`,
#             with the second GRU taking in outputs of the first GRU and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             GRU layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
#
class GRU(nn.GRU):

    def __init__(self, *,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 output_size=1,
                 steps=1,
                 activation=None,
                 activation_params=None,
                 batch_first=True,
                 **kwargs):
        #
        # Note: if output_size is <=0, no nn.Linear is created
        #
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)
        self.steps = steps
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f * hidden_size * steps, out_features=output_size)
        else:
            self.V = None
            self.output_size = f * hidden_size * steps

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]

        if L not in self.hidden:
            # D = self.D
            # N = self.hidden_size
            # hidden_state = torch.zeros(D, L, N)
            # self.hidden[L] = hidden_state
            self.hidden[L] = None

        hidden = self.hidden[L]
        t, h = super().forward(input, hidden)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        output = self.V(t) if self.V else t
        return output
# end


#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two RNNs together to form a `stacked RNN`,
#             with the second RNN taking in outputs of the first RNN and
#             computing the final results. Default: 1
#         nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             RNN layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
#
class RNN(nn.RNN):

    def __init__(self, *,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 output_size=1,
                 steps=1,
                 activation=None,
                 activation_params=None,
                 batch_first=True,
                 **kwargs):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)
        self._steps = steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f*hidden_size*steps, out_features=output_size)
        else:
            self.V = None
            self.output_size = f*hidden_size*steps

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]

        if L not in self.hidden:
            # D = self.D
            # N = self.hidden_size
            # hidden_state = torch.zeros(D, L, N)
            # self.hidden[L] = hidden_state
            self.hidden[L] = None

        hidden = self.hidden[L]
        t, h = super().forward(input, hidden)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        output = self.V(t) if self.V else t
        return output
# end
