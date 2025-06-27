from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BatchNorm1d
# ---------------------------------------------------------------------------
# Extends nn.BatchNorm1d to accept 'channels_last' parameter
#
#       (batch, channels, len)      channels_last=False
#       (batch, len, channels)      channels_last=True
#
#       [0,1,2] -> [0,2,1]
#

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            channels_last=True
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype
        )
        self.channels_last=channels_last
    # end

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
# BatchNorm2d
# ---------------------------------------------------------------------------
# Extends nn.BatchNorm2d to accept 'channels_last' parameter
#
#       (batch, channels, w, h)      channels_last=False
#       (batch, w, h, channels)      channels_last=True
#
#       [0,1,2,3] -> [0,2,3,1] -> [0,3,1,2]
#

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            channels_last=True
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype
        )
        self.channels_last = channels_last
    # end

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
