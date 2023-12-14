import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pow, sin, cos


# ---------------------------------------------------------------------------
# Snake activation function
# ---------------------------------------------------------------------------

class Snake(nn.Module):
    """
    Compute the Snake activation function. There are multiple definitions:

        'sin':    x + 1/a sin(a x)
        'cos':    x + 1/a cos(a x)
        'sin2':   x + 1/a sin(a x)^2
    """

    def __init__(self, a=1, method=None):
        super().__init__()
        self.a = float(a)
        self.method = method
        if method == 'sin':
            self._fun = lambda x: x + 1/a * sin(a*x)
        elif method == 'cos':
            self._fun = lambda x: x + 1/a * cos(a*x)
        elif method == 'sin2':
            self._fun = lambda x: x + 1/a * pow(sin(a*x), 2)
        else:
            self._fun = lambda x: x + 1/a * sin(a*x)

    def forward(self, x):
        return self._fun(x)
# end


# ---------------------------------------------------------------------------
# Non Negative Exponential Linear Unit activation function
# ---------------------------------------------------------------------------

ONE = torch.tensor(1, requires_grad=False)


class NNELU(nn.Module):
    """
    Compute the Non Negative Exponential Linear Unit:

        1 + elu(x)
    """

    def forward(self, x):
        return torch.add(F.elu(x), ONE)

