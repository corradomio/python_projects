#
# non-negative exponential linear unit
#
# https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
#
import torch
import torch.nn as nn
import torch.nn.functional as F

ONE = torch.tensor(1, requires_grad=False)


class NNELU(nn.Module):
    """
    Compute the Non Negative Exponential Linear Unit:

        1 + elu(x)
    """

    def forward(self, x):
        return torch.add(F.elu(x), ONE)
