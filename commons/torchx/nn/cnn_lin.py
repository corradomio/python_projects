import torch
import torch.nn as nn

from ..activation import activation_function
from ..utils import TorchLayerMixin


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
# in_channels: int,
#         out_channels: int,
#         kernel_size: _size_1_t,
#         stride: _size_1_t = 1,
#         padding: Union[str, _size_1_t] = 0,
#         dilation: _size_1_t = 1,
#         groups: int = 1,
#         bias: bool = True,
#         padding_mode: str = 'zeros',  # TODO: refine this type
#         device=None,
#         dtype=None
#
class Conv1dLinear(nn.Conv1d, TorchLayerMixin):
    def __init__(self, *,
                 input_size,
                 output_size,
                 hidden_size=1,
                 steps=1,
                 activation=None,
                 activation_params=None,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        super().__init__(in_channels=input_size,
                         out_channels=hidden_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups)

        self.activation = activation_function(activation, activation_params)
        self.lin = nn.Linear(in_features=hidden_size*steps, out_features=output_size)
    # end

    def forward(self, input):
        t = super().forward(input)
        t = self.activation(t) if self.activation else t
        t = torch.reshape(t, (len(input), -1))
        t = self.lin(t)
        return t
    # end
# end
