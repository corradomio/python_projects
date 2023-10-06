import torch
import torch.nn as nn

from ..utils import TorchLayerMixin


# ---------------------------------------------------------------------------
# Conv1d
# ---------------------------------------------------------------------------
# Extends nn.Conv1d to accept
#
#       (batch, channels, seq)      channel_last=False
#       (batch, seq, channels)      channel_last=True
#
#
class Conv1d(nn.Conv1d, TorchLayerMixin):
    def __init__(self, channels_last=False, **kwargs):
        super().__init__(**kwargs)
        self.channels_last = channels_last

    def forward(self, input):
        t = input
        if self.channels_last:
            t = torch.swapaxes(t,1, 2)

        t = super().forward(t)

        if self.channels_last:
            t = torch.swapaxes(t,1, 2)

        return t
    # end
# end
