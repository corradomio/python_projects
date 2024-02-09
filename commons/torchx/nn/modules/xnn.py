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

        or add new_dims dimensions at the end if n_dims > 0

            new_shape = T.shape + [1]*n_dims

        or at the begin if n_dims < 0

            new_shape = [1]*n_dims + T.shape

        :param shape: new tensor shape
        :param n_dims: new dimensions to add. If < 0, the dims are added in front
                otherwise at end
        """

        super().__init__()
        self.shape = list(shape) if shape else None
        self.n_dims = n_dims
        self.new_dims = [1]*n_dims if n_dims > 0 else [1]*(-n_dims)

    def forward(self, input: Tensor) -> Tensor:
        if self.shape is not None:
            shape = [input.shape[0]] + self.shape
        elif self.n_dims > 0:
            shape = list(input.shape) + self.new_dims
        else:
            shape = self.new_dims + list(input.shape)

        t = torch.reshape(input, shape)
        return t
# end


class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()
        if isinstance(shape, int):
            self.shape = (-1, shape)
        else:
            self.shape = (-1,) + tuple(shape)

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(x, self.shape)
# end


# ---------------------------------------------------------------------------
# RepeatVector
# ---------------------------------------------------------------------------
# As TF RepeatedVector
#

class RepeatVector(nn.Module):
    """
    Repeat a vector v in R^n, r times, generating a vector in R^(r,n)
    """

    def __init__(self, n_repeat=1):
        super().__init__()
        self.n_repeat = n_repeat

    def forward(self, x: Tensor) -> Tensor:
        n_repeat = self.n_repeat
        if n_repeat == 1:
            repeated = x
        else:
            repeated = x.repeat((n_repeat, 1))
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
            t = self.model(input[:, i, :])
            t_list.append(t)
        t = torch.cat(t_list, dim=1)

        rest_dims = list(t.shape[2:])

        new_dims = [t.shape[0], n_repeat, t.shape[1]//n_repeat] + rest_dims
        t = torch.reshape(t, shape=new_dims)
        return t
# end


# ---------------------------------------------------------------------------
# TimeRepeat
# ---------------------------------------------------------------------------

class TimeRepeat(nn.Module):

    def __init__(self, n_repeat=1, n_expand=0):
        """
        Repeat a 3D tensor (batch, seq, input) along 'input' dimension, generating a new 3D tensor
        with shape (batch, seq, r*input).

        The tensor can be 'expanded' with extra features, initialized to 0, if 'n_expand' is not 0.
        If 'n_expand' is greater than 0, it extends the tensor with zeros in front ([0...,  X])
        If 'n_expand' is less than 0, it extends the input tensor with zeros in back ([X, 0...])

        :param n_repeat: n of times the tensor is repeated along the 'input' dimension
        :param n_expand: if to add extra zeros in front (n_expand > 0) or at back (n_expand < 0)
            of the tensor.
        """
        super().__init__()
        self.n_repeat = n_repeat
        self.n_zeros = n_expand
        assert n_repeat > 0, "n_repeat needs to be an integer > 0"
        assert isinstance(n_expand, int), "n_zeros needs to be an integer"

    def forward(self, x: Tensor) -> Tensor:
        n_repeat = self.n_repeat
        n_zeros = self.n_zeros

        if n_zeros > 0:
            # add zeros in front
            shape = list(x.shape)
            shape[2] += n_zeros
            z = torch.zeros(shape)
            z[:, :, n_zeros:] += x
        elif n_zeros < 0:
            # add zeros at the back
            shape = list(x.shape)
            shape[2] += n_zeros
            z = torch.zeros(shape)
            z[:, :, :-n_zeros:] += x
        else:
            pass

        if n_repeat > 1:
            x = x.repeat(1, 1, n_repeat)

        return x
    # end
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
            y = self.model(input[:, :, i])
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
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------


