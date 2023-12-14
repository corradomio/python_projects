import torch
from torch import nn as nn
from torch import Tensor
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Loss function on tuples
# ---------------------------------------------------------------------------
# Compute the loss function on tuples
#

class MSELossTuple(nn.MSELoss):

    def forward(self, input: tuple[Tensor], target: tuple[Tensor]) -> Tensor:
        n = len(input)
        i0 = input[0]
        loss = torch.zeros((), dtype=i0.dtype, device=i0.device)
        for i in range(n):
            loss += F.mse_loss(input[i], target[i], reduction=self.reduction)
        return loss


class L1LossTuple(nn.L1Loss):

    def forward(self, input: tuple[Tensor], target: tuple[Tensor]) -> Tensor:
        n = len(input)
        i0 = input[0]
        loss = torch.zeros((), dtype=i0.dtype, device=i0.device)
        for i in range(n):
            loss += F.l1_loss(input[i], target[i], reduction=self.reduction)
        return loss
