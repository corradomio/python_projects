from typing import Union

import torch.nn as nn
from torch import Tensor

from ..utils import mul
from ..utils import TorchLayerMixin

# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
# It extends nn.Linear with an Integrated Flatten and Unflatten layers
#
#

class Linear(nn.Linear, TorchLayerMixin):

    def __init__(self,
                 in_features: Union[int, tuple[int, ...]],
                 out_features: Union[int, tuple[int, ...]],
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__(
            in_features=mul(in_features),
            out_features=mul(out_features),
            bias=bias,
            device=device,
            dtype=dtype
        )

        self.flatten = None if isinstance(in_features, int) \
            else nn.Flatten()
        self.unflatten = None if isinstance(out_features, int) \
            else nn.Unflatten(1, unflattened_size=out_features)

    def forward(self, input: Tensor) -> Tensor:
        t = input
        if self.flatten:
            t = self.flatten.forward(t)
        t = super().forward(t)
        if self.unflatten:
            t = self.unflatten.forward(t)
        return t
# end