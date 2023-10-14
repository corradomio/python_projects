from ... import nn as nnx


# tf.keras.layers.Conv1D(
#     filters, kernel_size, strides=1,
#     padding='valid',
#     data_format='channels_last',
#     dilation_rate=1,
#     groups=1,
#     activation=None,
#     use_bias=True,

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

class Conv1D(nnx.Conv1d):
    """
    Keras compatible Conv1D
    Note: the tensor shape is:

        (batch, seq, data_dim)

    """

    def __init__(self, input, filters,
                 strides=1, dilation_rate=1, use_bias=True,
                 channels_last=True,
                 **kwargs):
        super().__init__(
            in_channels=input,
            out_channels=filters,
            stride=strides,
            dilation=dilation_rate,
            bias=use_bias,
            channels_last=channels_last,     # invert the dimension orders
            **kwargs)

