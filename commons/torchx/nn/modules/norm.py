import torch.nn as nn
from torch import Tensor
from torch.nn.modules.normalization import _shape_t


class LayerNorm(nn.LayerNorm):
    # Add support for normalized_shape specified as a tuple
    # Note: it seems that it is already supported!

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)