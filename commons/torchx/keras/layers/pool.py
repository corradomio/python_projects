import torch
from torch import nn, Tensor


# tf.keras.layers.MaxPooling1D(
#     pool_size=2,
#     strides=None,
#     padding='valid',
#     data_format='channels_last',
#     **kwargs
# )
#
# torch.nn.MaxPool1d(
#     kernel_size: _size_1_t
#     stride: _size_1_t
#     padding: _size_1_t
#     dilation: _size_1_t.

class MaxPooling1D(nn.MaxPool1d):
    """
    Keras compatible MaxPooling1D

    Tensorflow MaxPooling1D is applied to the 3rd dimension
    Torch MaxPool1d is applied to the 2nd dimension
    to have the same behaviour the 2nd and 3rd axes are swapped
    """

    def __init__(self, pool_size=2, strides=None, **kwargs):
        super().__init__(kernel_size=pool_size, stride=strides, **kwargs)

    def forward(self, input: Tensor):
        assert len(input.shape) == 3
        t = torch.swapaxes(input, 1, 2)
        t = super().forward(t)
        t = torch.swapaxes(t, 1, 2)
        return t


class GlobalMaxPool1D(nn.Module):
    """
    Keras compatible GlobalMaxPool1D
    """

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, input: Tensor) -> Tensor:
        max_vals, max_idxs = torch.max(input, dim=1)
        t = max_vals
        if self.keepdims:
            t = t.reshape(list(t.shape) + [1])
        return t
