from typing import Tuple, Optional, Union

from torch import Tensor

from ... import nn as nnx


# tf.nn_keras.layers.LSTM(
#     units,
#     activation='tanh',
#     recurrent_activation='sigmoid',
#     use_bias=True,
#
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#
#     unit_forget_bias=True,
#
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#
#     dropout=0.0,
#     recurrent_dropout=0.0,
#
#     return_sequence=False,
#     return_state=False,
#
#     go_backwards=False,
#     stateful=False,
#     time_major=False,
#     unroll=False,
#     **kwargs
# )

#
# torch.nn.LSTM
#   weights:
#
#       all_weights = [
#           [weight_ih_l0,         weight_hh_l0,         bias_ih_l0,         bias_hh_l0        ],
#           [weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse]
#           ... 2 lists for each extra layer ...
#       ]
#
#       weight_ih_l0
#       weight_hh_l0
#       bias_ih_l0
#       bias_hh_l0
#
#       weight_ih_l0_reverse
#       weight_hh_l0_reverse
#       bias_ih_l0_reverse
#       bias_hh_l0_reverse
#
#       ... 8 members for each extra layer ...
#
#   note: l0 is the layer 0. If more layers are specified, there are
#

class LSTM(nnx.LSTM):
    """
    Keras compatible LSTM
    """

    # nn.LSTM returns a tuple:
    #
    #   1) the complete sequence
    #   2) a tuple containing the hidden state
    #
    # nn_keras.LSTM has 2 (+1) parameters that permits to specify what to return
    #
    #   1) return_sequence: if to return the complete sequence or just the last value
    #   2) return_state: if to return the hidden state
    #   3) it is possible to specify directly the activation function
    #

    def __init__(self, input, units,
                 return_sequence=True,
                 **kwargs):
        super().__init__(
            input_size=input,
            hidden_size=units,
            return_sequence=return_sequence,
            **kwargs)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tuple]]:

        state = None
        if self.return_state:
            seq, state = super().forward(input, hx)
        else:
            seq = super().forward(input, hx)

        if self.return_state:
            return seq, state
        else:
            return seq
# end


class GRU(nnx.GRU):
    """
    Keras compatible GRU
    """

    def __init__(self, input, units,
                 return_sequence=True,
                 **kwargs):
        super().__init__(
            input_size=input,
            hidden_size=units,
            return_sequence=return_sequence,
            **kwargs)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tuple]]:

        state = None
        if self.return_state:
            seq, state = super().forward(input, hx)
        else:
            seq = super().forward(input, hx)

        if self.return_state:
            return seq, state
        else:
            return seq
# end


class RNN(nnx.RNN):
    """
    Keras compatible RNN
    """

    def __init__(self, input, units,
                 return_sequence=True,
                 **kwargs):
        super().__init__(
            input_size=input,
            hidden_size=units,
            return_sequence=return_sequence,
            **kwargs)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tuple]]:

        state = None
        if self.return_state:
            seq, state = super().forward(input, hx)
        else:
            seq = super().forward(input, hx)

        if self.return_state:
            return seq, state
        else:
            return seq
# end
