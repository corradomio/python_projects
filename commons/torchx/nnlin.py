from typing import Union, Optional

import torch
import torch.nn as nn
from stdlib import is_instance
from . import nn as nnx

from .nn_init import activation_function


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class LinearEncoderDecoder(nn.Module):

    def __init__(self, *,
                 input_size: Union[int, tuple[int, int]],
                 output_size: Union[int, tuple[int, int]],
                 hidden_size: Union[None, int, tuple[int, int]] = None,
                 activation: Optional[str] = None,
                 activation_kwargs: Optional[dict] = None):
        super().__init__()

        assert is_instance(input_size, Union[int, tuple[int, int]])
        assert is_instance(output_size, Union[int, tuple[int, int]])
        assert is_instance(hidden_size, Union[None, int, tuple[int, int]])
        assert is_instance(activation, Optional[str])
        assert is_instance(activation_kwargs, Optional[dict])

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        if hidden_size is None:
            self.lin = nnx.Linear(in_features=input_size, out_features=output_size)
        else:
            self.enc = nnx.Linear(in_features=input_size, out_features=hidden_size)
            self.activation = activation_function(activation, activation_kwargs)
            self.dec = nnx.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        if self.hidden_size is None:
            return self.lin.forward(x)
        else:
            t = self.enc(x)
            t = self.activation(t)
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
                 input_shape,
                 output_shape,
                 # LSTM specific
                 hidden_size=1,
                 num_layers=1,
                 activation=None,
                 activation_kwargs=None,
                 batch_first=True,
                 **kwargs):
        """
        A simple NN model composed by a LSTM layer followed bay a Linear layer.
        The parameters 'input_shape' and 'output_shape' are used to specify:

            input_shape: (n_steps, input_size|n_features)
            ouput_shape: (n_steps, output_size)

        :param input_shape:
        :param output_shape:
        :param hidden_size:
        :param num_layers:
        :param activation:
        :param activation_kwargs:
        :param batch_first:
        :param kwargs:
        """

        assert is_instance(input_shape, tuple[int, int])
        assert is_instance(output_shape, tuple[int, int])
        assert is_instance(hidden_size, int)

        super().__init__(input_size=input_shape[1],
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape

        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        steps = input_shape[0]
        self.activation = activation_function(activation, activation_kwargs)
        self.lin = nnx.Linear(in_features=f * hidden_size * steps, out_features=output_shape)
        return

    def forward(self, input, hx=None):
        # t, h = super().forward(input, hx)
        t = super().forward(input, hx)
        t = self.activation(t)
        # t = torch.reshape(t, (len(input), -1))
        output = self.lin(t) if self.lin else t
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
                 input_shape,
                 output_shape,
                 # GRU specific
                 hidden_size=1,
                 num_layers=1,
                 activation=None,
                 activation_kwargs=None,
                 batch_first=True,
                 **kwargs):
        """
        A simple NN model composed by a GRU layer followed bay a Linear layer.
        The parameters 'input_shape' and 'output_shape' are used to specify:

            input_shape: (n_steps, input_size|n_features)
            ouput_shape: (n_steps, output_size)

        :param input_shape:
        :param output_shape:
        :param hidden_size:
        :param num_layers:
        :param activation:
        :param activation_kwargs:
        :param batch_first:
        :param kwargs:
        """
        assert is_instance(input_shape, tuple[int, int])
        assert is_instance(output_shape, tuple[int, int])
        assert is_instance(hidden_size, int)
        #
        # Note: if output_size is <=0, no nn.Linear is created
        #
        super().__init__(input_size=input_shape[1],
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape

        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        steps = input_shape[0]
        self.activation = activation_function(activation, activation_kwargs)
        self.lin = nnx.Linear(in_features=f * hidden_size * steps, out_features=output_shape)
        return

    def forward(self, input, hx=None):
        # t, h = super().forward(input, hx)
        t = super().forward(input, hx)
        t = self.activation(t)
        # t = torch.reshape(t, (len(input), -1))
        output = self.lin(t) if self.lin else t
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
                 input_shape,
                 output_shape,
                 # RNN specific
                 hidden_size,
                 num_layers=1,
                 activation=None,
                 activation_kwargs=None,
                 batch_first=True,
                 **kwargs):
        assert is_instance(input_shape, tuple[int, int])
        assert is_instance(output_shape, tuple[int, int])
        assert is_instance(hidden_size, int)

        super().__init__(input_size=input_shape[1],
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         **kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape

        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers

        steps = input_shape[0]
        self.activation = activation_function(activation, activation_kwargs)
        self.lin = nnx.Linear(in_features=f * hidden_size * steps, out_features=output_shape)
        return

    def forward(self, input, hx=None):
        t = super().forward(input, hx)
        t = self.activation(t)
        # t = torch.reshape(t, (len(input), -1))
        output = self.lin(t) if self.lin else t
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
class Conv1dLinear(nnx.Conv1d):
    def __init__(self, *,
                 input_size,
                 output_size,
                 hidden_size=1,
                 # steps=1,
                 activation=None,
                 activation_kwargs=None,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 **kwargs):
        assert is_instance(input_size, tuple[int, int])
        assert is_instance(output_size, tuple[int, int])
        assert is_instance(hidden_size, int)

        super().__init__(in_channels=input_size[1],
                         out_channels=hidden_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         **kwargs)

        steps = input_size[0]
        self.activation = activation_function(activation, activation_kwargs)

        self.lin = nnx.Linear(in_features=steps * hidden_size, out_features=output_size)
        self.output_size = output_size
    # end

    def forward(self, input):
        t = super().forward(input)
        t = self.activation(t)
        # t = torch.reshape(t, (len(input), -1))
        t = self.lin(t) if self.lin else t
        return t
    # end
# end


# ---------------------------------------------------------------------------
# MultiInputs
# ---------------------------------------------------------------------------

# class MultiInputs(nn.Module):
#
#     def __init__(self, input_models, output_model):
#         super().__init__()
#
#         self.input_models = input_models
#         self.output_model = output_model
#         self.n_inputs = len(input_models)
#
#     def forward(self, input_list):
#         assert len(input_list) == self.n_inputs
#         n = self.n_inputs
#         inner_results = []
#
#         # for each input, call the related models
#         for i in range(n):
#             input = input_list[i]
#             model = self.input_models[i]
#
#             inner_result = model(input)
#             inner_results.append(inner_result)
#
#         # concatenate the inner results
#         inner = torch.concatenate(inner_results, dim=1)
#
#         result = self.output_model(inner)
#         return result
#     # end
# # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
