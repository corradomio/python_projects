from typing import Union

from torch import Tensor

from ...utils import KerasLayerMixin
from ... import nn as nnx
from ...activation import activation_function


# ---------------------------------------------------------------------------
# Dense Layer
# ---------------------------------------------------------------------------
# Mix of nn.Layer and keras.Dense
#
#       1) it is possible to specify the input/unit dimensions as a tuple
#       2) it is possible to specify the activation function

# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

class Dense(nnx.Linear, KerasLayerMixin):
    def __init__(self,
                 input: Union[int, tuple],
                 units: Union[int, tuple],
                 activation=None,
                 **kwargs):
        super().__init__(in_features=input, out_features=units, **kwargs)
        self.activation = activation_function(activation)

    def forward(self, input: Tensor) -> Tensor:
        t = super().forward(input)
        if self.activation:
            t = self.activation.forward(t)
        return t


# class Dense(nn.Sequential):
#     def __init__(self,
#                  input: Union[int, tuple],
#                  units: Union[int, tuple],
#                  bias: bool = True,
#                  device=None,
#                  dtype=None,
#                  activation=None):
#
#         if isinstance(input, int) and isinstance(units, int):
#             super().__init__(
#                 nn.Linear(
#                     in_features=input,
#                     out_features=units,
#                     bias=bias,
#                     device=device,
#                     dtype=dtype
#                 )
#             )
#         else:
#             in_features: list[int] = ranked(input)
#             out_features: list[int] = ranked(units)
#
#             super().__init__(
#                 nn.Flatten(),
#                 nn.Linear(
#                     in_features=mul(in_features),
#                     out_features=mul(out_features),
#                     bias=bias,
#                     device=device,
#                     dtype=dtype
#                 ),
#                 nn.Unflatten(1, unflattened_size=out_features)
#             )
#
#         self.activation = activation_function(activation)
#
#     def forward(self, input: Tensor) -> Tensor:
#         t = super().forward(input)
#         if self.activation:
#             t = self.activation.forward(t)
#         return t
