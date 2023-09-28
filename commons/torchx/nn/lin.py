import torch.nn as nn
from typing import Union
from torch import Tensor
from ..utils import ranked, mul


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
# It extends nn.Linear with an Integrated Flatten and Unflatten layers
#

class Linear(nn.Linear):

    def __init__(self,
                 in_features: Union[int, tuple],
                 out_features: Union[int, tuple],
                 bias: bool = True,
                 device=None,
                 dtype=None):
        in_features: list[int] = ranked(in_features)
        out_features: list[int] = ranked(out_features)
        super().__init__(
            in_features=mul(in_features),
            out_features=mul(out_features),
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, unflattened_size=out_features)

    def forward(self, input: Tensor) -> Tensor:
        t = self.flatten.forward(input)
        t = super().forward(t)
        t = self.unflatten.forward(t)
        return t
# end
