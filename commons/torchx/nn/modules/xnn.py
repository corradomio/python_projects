import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# ReshapeVector
# ---------------------------------------------------------------------------
# It is equivalent to nn.Unflatten(...) but more simple

class ReshapeVector(nn.Module):
    def __init__(self, shape=None, n_dims=0):
        """
        Reshape the input tensor T as

            new_shape = T.shape[0] + shape

        or add new_dims dimensions as

            new_shape = T.shape = [1 for i in range(new_dims)]

        :param shape: new tensor shape
        :param n_dims: new dimensions to add. If < 0, the dims are added in front
                otherwise at end
        """

        super().__init__()
        self.shape = list(shape) if shape else None
        self.suffix = n_dims > 0
        self.n_dims = [1 for i in range(abs(n_dims))]

    def forward(self, input: Tensor) -> Tensor:
        if self.shape:
            shape = [input.shape[0]] + self.shape
        elif self.suffix:
            shape = list(input.shape) + self.n_dims
        else:
            shape = self.n_dims + list(input.shape)
        t = torch.reshape(input, shape)
        return t


# ---------------------------------------------------------------------------
# RepeatVector
# ---------------------------------------------------------------------------
# As TF RepeatedVector
#

class RepeatVector(nn.Module):

    def __init__(self, n_repeat=1):
        super().__init__()
        self.n_repeat = n_repeat

    def forward(self, input: torch.Tensor):
        rep_list = [input for i in range(self.n_repeat)]
        repeated = torch.stack(rep_list, 1)
        return repeated
# end


# ---------------------------------------------------------------------------
# TimeDistributed
# ---------------------------------------------------------------------------
# As TF TimeDistributed
#
#   RNN:            (batch, seq, input)   ->  (batch, seq, units)
#   Distributed:    (batch, seq, units)   ->  (batch, seq,     1)
#

class TimeDistributed(nn.Module):

    def __init__(self, *models):
        super().__init__()

        assert len(models) > 0, "TimeDistributed needs 1+ models to apply"

        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = nn.Sequential(*models)

    def forward(self, input):
        n_repeat = input.shape[1]

        t_list = []
        for i in range(n_repeat):
            t = self.model.forward(input[:, i, :])
            t_list.append(t)
        t = torch.cat(t_list, dim=1)

        rest_dims = list(t.shape[2:])

        new_dims = [t.shape[0], n_repeat, t.shape[1]//n_repeat] + rest_dims
        t = torch.reshape(t, shape=new_dims)
        return t
# end


# ---------------------------------------------------------------------------
# ChannelDistributed
# ---------------------------------------------------------------------------
# As TF ChannelDistributed (IF it exists)
#
#   CNN:            (batch, input, seq)   ->  (batch, units, seq)
#   Distributed:    (batch, seq, units)   ->  (batch, 1,     seq)
#

class ChannelDistributed(nn.Module):

    def __init__(self, *models):
        super().__init__()

        assert len(models) > 0, "ChannelDistributed needs 1+ models to apply"

        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = nn.Sequential(*models)

    def forward(self, input):
        n_repeat = input.shape[2]

        y_list = []
        for i in range(n_repeat):
            y = self.model.forward(input[:, :, i])
            y_list.append(y)

        out = torch.cat(y_list, dim=1)
        out = torch.reshape(out, shape=(out.shape[0], out.shape[1]//n_repeat, n_repeat))
        return out
# end


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

class Clip(nn.Module):

    def __init__(self, clip=(0, 1)):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        cmin, cmax = self.clip
        x[x < cmin] = cmin
        x[x > cmax] = cmax
        return x


