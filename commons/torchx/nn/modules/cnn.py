from typing import Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t


# ---------------------------------------------------------------------------
# Conv1d
# ---------------------------------------------------------------------------
# Extends nn.Conv1d to accept 'channel_last' parameter
#
#       (batch, channels, seq)      channel_last=False
#       (batch, seq, channels)      channel_last=True
#
#
class Conv1d(nn.Conv1d):
    """
    Extends nn.Conv1d to accept 'channel_last' parameter

    Args:
        kernel_size: Default 1
        channels_last: If ``True``, the channels are located as in RNN

            (batch, seq, channels)

    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t = 1,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None,
                 channels_last=True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.channels_last = channels_last

    def forward(self, input):
        t = input
        if self.channels_last:
            t = torch.swapaxes(t, 1, 2)
            t = super().forward(t)
            t = torch.swapaxes(t, 1, 2)
        else:
            t = super().forward(t)

        return t
    # end
# end


# ---------------------------------------------------------------------------
# create_cnn
# ---------------------------------------------------------------------------

CNNX_FLAVOURS = {
    'cnn': Conv1d,
}

CNNX_PARAMS = [
    'in_channels', 'out_channels',
    'kernel_size', 'stride', 'dilation',
    'groups', 'bias',
    'padding', 'padding_mode',
    'device', 'dtype',
    # extended parameters
    'channels_last'
]


def create_cnn(flavour: str, **kwargs):
    if flavour not in CNNX_FLAVOURS:
        raise ValueError(f"Unsupported CNN flavour {flavour}")
    else:
        return CNNX_FLAVOURS[flavour](**kwargs)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
