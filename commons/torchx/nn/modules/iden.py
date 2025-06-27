import torch.nn as nn


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
# Simple layer do nothing.
# Useful as place holder

class Identity(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, *args):
        return args
# end

# ---------------------------------------------------------------------------
# ENd
# ---------------------------------------------------------------------------
