import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PowerModule
# ---------------------------------------------------------------------------

class PowerModule(nn.Module):

    def __init__(self, order: int = 1, cross: int = 1):
        self.order = order
        self.cross = cross

    def forward(self, X):
        if self.order == 1:
            return X
        Xcat = []
        for i in range(1, self.order+1):
            Xi = torch.pow(X, i)
            Xcat.append(Xi)
        return torch.cat(Xcat, 1)
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
