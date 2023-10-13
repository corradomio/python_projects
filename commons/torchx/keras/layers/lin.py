from typing import Union

from ... import nn as nnx


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

class Dense(nnx.Linear):
    """
    Keras compatible Dense
    """

    def __init__(self,
                 input: Union[int, tuple],
                 units: Union[int, tuple],
                 **kwargs):
        super().__init__(in_features=input, out_features=units, **kwargs)
