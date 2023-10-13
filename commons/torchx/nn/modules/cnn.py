import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Conv1d
# ---------------------------------------------------------------------------
# Extends nn.Conv1d to accept
#
#       (batch, channels, seq)      channel_last=False
#       (batch, seq, channels)      channel_last=True
#
#
class Conv1d(nn.Conv1d):
    """
    Extends nn.Conv1d

    Args:
        kernel_size: Default 1
        channels_last: If ``True``, the channels are located as in RNN

            (batch, seq, channels)
    """
    def __init__(self,
                 kernel_size=1,
                 channels_last=False,
                 **kwargs):
        super().__init__(kernel_size=kernel_size, **kwargs)
        self.channels_last = channels_last

    def forward(self, input):
        t = input
        if self.channels_last:
            t = torch.swapaxes(t, 1, 2)

        t = super().forward(t)

        if self.channels_last:
            t = torch.swapaxes(t, 1, 2)

        return t
    # end
# end
