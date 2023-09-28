from typing import Tuple, Optional, Union

import torch.nn as nn
from torch import Tensor
from ...activation import activation_function


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
#     return_sequences=False,
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

class LSTM(nn.LSTM):
    # nn.LSTM returns a tuple:
    #
    #   1) the complete sequence
    #   2) a tuple containing the hidden state
    #
    # nn_keras.LSTM has 2 parameters that permits to specify what to return
    #
    #   1) return_sequence: if to return the complete sequence or just the
    #      last value
    #   2) return_state: if to return the hidden state
    #
    # it is also possible

    def __init__(self,
                 input=None,
                 units=None,
                 return_sequence=True,
                 return_state=False,
                 activation=None,
                 **kwargs):
        super().__init__(input_size=input, hidden_size=units, **kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.activation = activation
        self._af = activation_function(activation)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tuple]]:
        seq, state = super().forward(input, hx)
        if self._af:
            seq = self._af.forward(seq)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        else:
            return seq[:, -1]
