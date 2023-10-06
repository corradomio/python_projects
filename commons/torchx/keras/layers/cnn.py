from torch import Tensor

from ...utils import KerasLayerMixin
from ... import nn as nnx
from ...activation import activation_function


# tf.keras.layers.Conv1D(
#     filters, kernel_size, strides=1,
#     padding='valid',
#     data_format='channels_last',
#     dilation_rate=1,
#     groups=1,
#     activation=None,
#     use_bias=True,
#
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# torch.nn.Conv1d(
#     in_channels: int,
#     out_channels: int,
#     kernel_size: _size_1_t,
#     stride: _size_1_t = 1,
#     dilation: _size_1_t = 1,
#
#     padding: Union[str, _size_1_t] = 0,
#     padding_mode: str = 'zeros',  # TODO: refine this type
#     groups: int = 1,
#     bias: bool = True,
#     device=None,
#     dtype=None
# ).

# tf padding:
#   valid   means no padding
#   same    padding with zeros
#   causal  causal (dilated) convolutions
#
# torch padding

class Conv1D(nnx.Conv1d, KerasLayerMixin):

    def __init__(self, *args,
                 activation=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation_function(activation)

    def forward(self, input: Tensor) -> Tensor:
        t = super().forward(input)
        if self.activation:
            t = self.activation.forward(t)
        return t

