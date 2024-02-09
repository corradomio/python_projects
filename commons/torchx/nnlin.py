import torch
import torch.nn as nn
from . import nn as nnx

from .activation import activation_function


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class LinearEncoderDecoder(nn.Module):

    def __init__(self, *,
                 input_shape,
                 output_shape,
                 hidden_size=None,
                 activation=None,
                 activation_params=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size

        if hidden_size is None:
            self.lin = nnx.Linear(in_features=input_shape, out_features=output_shape)
        else:
            self.enc = nnx.Linear(in_features=input_shape, out_features=hidden_size)
            self.activation = activation_function(activation, activation_params)
            self.dec = nnx.Linear(in_features=hidden_size, out_features=output_shape)

    def forward(self, x):
        if self.hidden_size is None:
            return self.lin.forward(x)
        else:
            t = self.enc(x)
            t = self.activation(t) if self.activation else t
            return self.dec(t)
# end


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
class LSTMLinear(nnx.LSTM):

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
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f * hidden_size * steps, out_features=output_size)
            self.output_size = output_size
        else:
            self.V = None
            self.output_size = f * hidden_size * steps

    def forward(self, input, hx=None):
        t, h = super().forward(input, hx)
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
class GRULinear(nnx.GRU):

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
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f * hidden_size * steps, out_features=output_size)
            self.output_size = output_size
        else:
            self.V = None
            self.output_size = f * hidden_size * steps

    def forward(self, input, hx=None):
        t, h = super().forward(input, hx)
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
class RNNLinear(nnx.RNN):

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
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.V = nn.Linear(in_features=f*hidden_size*steps, out_features=output_size)
            self.output_size = output_size
        else:
            self.V = None
            self.output_size = f*hidden_size*steps

    def forward(self, input, hx=None):
        t, h = super().forward(input, hx)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        output = self.V(t) if self.V else t
        return output
# end


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
# in_channels: int,
#         out_channels: int,
#         kernel_size: _size_1_t,
#         stride: _size_1_t = 1,
#         padding: Union[str, _size_1_t] = 0,
#         dilation: _size_1_t = 1,
#         groups: int = 1,
#         bias: bool = True,
#         padding_mode: str = 'zeros',  # TODO: refine this type
#         device=None,
#         dtype=None
#
class Conv1dLinear(nn.Conv1d):
    def __init__(self, *,
                 input_size,
                 output_size,
                 hidden_size=1,
                 steps=1,
                 activation=None,
                 activation_params=None,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        super().__init__(in_channels=input_size,
                         out_channels=hidden_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups)

        self.activation = activation_function(activation, activation_params)
        if output_size > 0:
            self.lin = nn.Linear(in_features=hidden_size * steps, out_features=output_size)
            self.output_size = output_size
        else:
            self.lin = None
            self.output_size = hidden_size * steps
    # end

    def forward(self, input):
        t = super().forward(input)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        t = self.lin(t) if self.lin else t
        return t
    # end
# end


# ---------------------------------------------------------------------------
# MultiInputs
# ---------------------------------------------------------------------------

class MultiInputs(nn.Module):

    def __init__(self, input_models, output_model):
        super().__init__()

        self.input_models = input_models
        self.output_model = output_model
        self.n_inputs = len(input_models)

    def forward(self, input_list):
        assert len(input_list) == self.n_inputs
        n = self.n_inputs
        inner_results = []

        # for each input, call the related models
        for i in range(n):
            input = input_list[i]
            model = self.input_models[i]

            inner_result = model(input)
            inner_results.append(inner_result)

        # concatenate the inner results
        inner = torch.concatenate(inner_results, dim=1)

        result = self.output_model(inner)
        return result
    # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
