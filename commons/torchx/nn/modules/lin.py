from typing import Union

import torch.nn as nn
from torch import Tensor


def mul(dim: Union[int, list[int]]) -> int:
    if isinstance(dim, int):
        return dim
    d = 1
    for l in dim:
        d *= l
    return d


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
# It extends nn.Linear with an Integrated Flatten and Unflatten layers
#
#

class Linear(nn.Linear):
    """
    Extends nn.Linear

    It applies flatten/unflatten if ``in_features``/``out_features``
    are not integers.

    Args:
        in_features: can be a tuple
        out_features: can be a tuple
    """

    def __init__(self,
                 in_features: Union[int, tuple[int, ...]] = None,
                 out_features: Union[int, tuple[int, ...]] = None,
                 **kwargs):

        super().__init__(
            in_features=mul(in_features),
            out_features=mul(out_features),
            **kwargs)

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
