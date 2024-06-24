
__all__ = [
    'TemporalConvNetwork'
]

from typing import Union
import torch.nn as nn
from torch.nn.utils import weight_norm

# ---------------------------------------------------------------------------
# https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks
# ---------------------------------------------------------------------------
# Based on https://github.com/locuslab/TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: Union[int, list[int]],
                 kernel_size=2,
                 dropout=0.2,
                 channels_last=False):
        super(TemporalConvNetwork, self).__init__()
        layers = []

        # Note: 'out_channels' must be a list containing the number of hidden channels for each level
        # BUT the number of levels depends on the sequence length
        if isinstance(out_channels, int):
            out_channels = [out_channels]

        n_levels = len(out_channels)
        for i in range(n_levels):
            dilation_size = 2 ** i
            in_channels_level = in_channels if i == 0 else out_channels[i - 1]
            out_channels_level = out_channels[i]
            layers += [TemporalBlock(in_channels_level, out_channels_level,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.channels_last = channels_last

    def forward(self, x):
        t = x
        if self.channels_last:
            t = t.swapaxes(1, 2)

        t = self.network(t)

        if self.channels_last:
            t = t.swapaxes(1, 2)
        return t

