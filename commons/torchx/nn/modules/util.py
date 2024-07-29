import torch
import torch.nn as nn
from torch import Tensor
from .module import Module


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
#

def print_shape(what, x, i=0):
    if isinstance(x, (list, tuple)):
        if i == 0:
            print("  "*i, what, "...")
        else:
            print("  " * i, "...")
        for t in x:
            print_shape(what, t, i+1)
        return
    if i == 0:
        print("  "*i, what, tuple(x.shape))
    else:
        print("  " * i, tuple(x.shape))


class Probe(Module):
    """
    Used to insert breakpoints during the training/prediction and to print the
    tensor shapes (ONLY the first time)
    """

    def __init__(self, name="probe"):
        super().__init__()
        self.name = name
        self._log = True
        self._repr = f"[{name}]\t"

    def forward(self, input):
        if self._log:
            print_shape(self._repr, input)
            self._log = False
        return input

    def __repr__(self):
        return self._repr
# end


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------
#

class Select(Module):
    """
    If  'input' is a tuple/list o an hierarchical structure, it permits to
    select an element based on a sequence of indices:

        select=(3,2,4,1)

    is converted into

        input[3][2][4][1]
    """

    def __init__(self, select=()):
        super().__init__()
        assert isinstance(select, (int, list, tuple))
        self.select = [select] if isinstance(select, int) else list(select)

    def forward(self, input):
        output = input
        for s in self.select:
            output = output[s]
        return output
# end


# ---------------------------------------------------------------------------
# ReshapeVector
# ---------------------------------------------------------------------------
# It is equivalent to nn.Unflatten(...) but more simple

class ReshapeVector(Module):
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


class Reshape(Module):

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

class RepeatVector(Module):
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

class TimeDistributed(Module):

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

class ChannelDistributed(Module):

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

class Clip(Module):

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
