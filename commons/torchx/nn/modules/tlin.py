import torch.nn as nn
from torch import Tensor
from ...utils import time_repeat


class TimeLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 replicate: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        assert isinstance(in_features, int), "in_features"
        assert isinstance(out_features, int), "out_features"
        assert isinstance(replicate, int), "replicate"

        self.input_shape = in_features
        self.output_shape = out_features
        self.replicate = replicate
    # end

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2, "It is not a 2D tensor"
        t = super().forward(x)
        y = time_repeat(t, self.replicate)
        return y
    # end
# end
