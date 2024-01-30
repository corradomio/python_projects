from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from stdlib import mul_
from ...utils import time_repeat, is_shape


class ZeroCache(nn.Module):

    def __init__(self, feature_shape):
        super().__init__()
        assert is_shape(feature_shape)
        self.feature_shape = feature_shape
        self._zero_cache = dict()

    def forward(self, x):
        batch_size = len(x)
        if batch_size not in self._zero_cache:
            zeros = torch.zeros((batch_size, ) + self.feature_shape,
                               dtype=x.dtype, device=x.device, requires_grad=False)
            self._zero_cache[batch_size] = zeros
        return self._zero_cache[batch_size]
    # end
# end


class TimeLinear(nn.Linear):
    def __init__(self,
                 in_features: Union[int, tuple[int, int]],
                 out_features: tuple[int, int],
                 bias: bool = True, device=None, dtype=None):
        assert is_shape(in_features), "in_features"
        assert is_shape(in_features), "out_features"
        super().__init__(
            in_features=mul_(in_features),
            out_features=out_features[1],
            bias=bias,
            device=device,
            dtype=dtype
        )

        self.input_shape = in_features
        self.output_shape = out_features
        self.replicate = out_features[0]
    # end

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) > 2:
            batch_size = len(x)
            x = torch.reshape(x, (batch_size, -1))
        t = super().forward(x)
        y = time_repeat(t, self.replicate)
        return y
    # end
# end


def apply_if(t, transformer):
    return t if transformer is None else transformer(t)