from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from stdlib import mul_


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
# It extends nn.Linear with an integrated Flatten and Unflatten layers
#

class Linear(nn.Linear):
    """
    Extends nn.Linear to accept ``in_feature`` and ``out_features`` as tuples.
    If a parameter is a tuple, it applies a flatten/unflatten transformation

    Args:
        in_features: can be a tuple
        out_features: can be a tuple
    """

    def __init__(self,
                 in_features: Union[int, tuple[int, ...]],
                 out_features: Union[int, tuple[int, ...]],
                 bias: bool = True, device=None, dtype=None):

        super().__init__(
            in_features=mul_(in_features),
            out_features=mul_(out_features),
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.input_shape = in_features
        self.output_shape = out_features

        # self.flatten = None if isinstance(in_features, int) \
        #     else nn.Flatten()
        # self.unflatten = None if isinstance(out_features, int) \
        #     else nn.Unflatten(1, unflattened_size=out_features)

    def forward(self, input: Tensor) -> Tensor:
        t = input

        # if not isinstance(self.input_shape, int):
        #     t = torch.flatten(t, start_dim=1)
        if len(input.shape) > 2:
            t = torch.flatten(t, start_dim=1)

        t = super().forward(t)

        if not isinstance(self.output_shape, int):
            # t = t.view((-1,) + self.output_shape)
            t = torch.reshape(t, (-1,) + self.output_shape)

        return t
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
