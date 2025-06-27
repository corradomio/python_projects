from typing import Optional

import torch
import torch.nn as nn
from torch.nn.common_types import _size_any_t


# ---------------------------------------------------------------------------
# MaxPool1d
# ---------------------------------------------------------------------------
# Extends nn.MaxPool1d to accept 'channels_last' parameter
#
#       (batch, channels, len)      channels_last=False
#       (batch, len, channels)      channels_last=True
#
#       [0,1,2] -> [0,2,1]
#

class MaxPool1d(nn.MaxPool1d):
    def __init__(
            self,
            kernel_size: _size_any_t,
            stride: Optional[_size_any_t] = None,
            padding: _size_any_t = 0,
            dilation: _size_any_t = 1,
            return_indices: bool = False,
            ceil_mode: bool = False,
            channels_last=True
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.channels_last = channels_last
    # end

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
# MaxPool2d
# ---------------------------------------------------------------------------
# Extends nn.MaxPool2d to accept 'channels_last' parameter
#
#       (batch, channels, w, h)      channels_last=False
#       (batch, w, h, channels)      channels_last=True
#
#       [0,1,2,3] -> [0,2,3,1] -> [0,3,1,2]
#

class MaxPool2d(nn.MaxPool2d):
    def __init__(
            self,
            kernel_size: _size_any_t,
            stride: Optional[_size_any_t] = None,
            padding: _size_any_t = 0,
            dilation: _size_any_t = 1,
            return_indices: bool = False,
            ceil_mode: bool = False,
            channels_last=True
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.channels_last = channels_last
    # end

    def forward(self, input):
        t = input
        if self.channels_last:
            t = torch.permute(t, [0, 3, 1, 2])
            t = super().forward(t)
            t = torch.permute(t, [0, 2, 3, 1])
        else:
            t = super().forward(t)

        return t
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
