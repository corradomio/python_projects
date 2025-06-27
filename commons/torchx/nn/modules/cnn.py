from typing import Union, Optional

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t


# ---------------------------------------------------------------------------
# Conv1d
# ---------------------------------------------------------------------------
# Extends nn.Conv1d to accept 'channels_last' parameter
#
#       (batch, channels, len)      channels_last=False
#       (batch, len, channels)      channels_last=True
#
#       [0,1,2] -> [0,2,1]
#
class Conv1d(nn.Conv1d):
    """
    Extends nn.Conv1d to accept 'channels_last' parameter

    Args:
        kernel_size: Default 1
        channels_last: If ``True``, the channels are located as in RNN

            (batch, len, channels)

    """
    def __init__(
            self,
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
            channels_last=True
    ):
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
# Conv2d
# ---------------------------------------------------------------------------
# Extends nn.Conv2d to accept 'channels_last' parameter
#
#       (batch, channels, w, h)      channels_last=False
#       (batch, w, h, channels)      channels_last=True
#
#       [0,1,2,3] -> [0,2,3,1] -> [0,3,1,2]
#
class Conv2d(nn.Conv2d):
    """
    Extends nn.Conv1d to accept 'channels_last' parameter

    Args:
        kernel_size: Default 1
        channels_last: If ``True``, the channels are located as in RNN

            (batch, seq, channels)

    """
    def __init__(
            self,
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
            channels_last=True
    ):
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
            t = torch.permute(t, [0,3,1,2])
            t = super().forward(t)
            t = torch.permute(t, [0,2,3,1])
        else:
            t = super().forward(t)

        return t
    # end
# end


# ---------------------------------------------------------------------------
# ConvTranspose1d
# ---------------------------------------------------------------------------
# Extends nn.ConvTranspose1d to accept 'channels_last' parameter
#
#       (batch, channels, len)      channels_last=False
#       (batch, len, channels)      channels_last=True
#
#       [0,1,2] -> [0,2,1]
#
class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: _size_1_t = 0,
            output_padding: _size_1_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_1_t = 1,
            padding_mode: str = "zeros",
            device=None,
            dtype=None,
            channels_last=True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.channels_last=channels_last

    def forward(self, input: torch.Tensor, output_size: Optional[list[int]] = None) -> torch.Tensor:
        t = input
        if self.channels_last:
            t = torch.swapaxes(t, 1, 2)
            t = super().forward(t, output_size)
            t = torch.swapaxes(t, 1, 2)
        else:
            t = super().forward(t)

        return t
    # end
# end


# ---------------------------------------------------------------------------
# ConvTranspose2d
# ---------------------------------------------------------------------------
# Extends nn.ConvTranspose1d to accept 'channels_last' parameter
#
#       (batch, channels, w, h)      channels_last=False
#       (batch, w, h, channels)      channels_last=True
#
#       [0,1,2,3] -> [0,2,3,1] -> [0,3,1,2]
#
class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            output_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_2_t = 1,
            padding_mode: str = "zeros",
            device=None,
            dtype=None,
            channels_last=True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.channels_last = channels_last

    def forward(self, input: torch.Tensor, output_size: Optional[list[int]] = None) -> torch.Tensor:
        t = input
        if self.channels_last:
            t = torch.permute(t, [0, 3, 1, 2])
            t = super().forward(t, output_size)
            t = torch.permute(t, [0, 2, 3, 1])
        else:
            t = super().forward(t)

        return t
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
