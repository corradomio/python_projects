from typing import Optional

import torch.nn as nn
from numpy.typing import ArrayLike
from torch import Tensor

from .ts import TimeSeriesModel
from .tsutils import apply_if
from ..modules.tcn import TCN


class TSTCN(TimeSeriesModel):

    def __init__(self,
                 input_shape, output_shape,
                 feature_size=None,
                 num_channels: Optional[list[int]] = None,  # num of blocks = len(num_channels)
                 kernel_size: int = 4,
                 dilations: Optional[ArrayLike] = None,
                 dilation_reset: Optional[int] = None,
                 dropout: float = 0.1,
                 use_norm: str = 'weight_norm',
                 activation: str = 'relu',
                 use_skip_connections: bool = False,
                 causal: bool = False,
                 # kernel_initializer: str = 'xavier_uniform',
                 # input_type: str = 'NCL',
                 ):
        super().__init__(
            input_shape=input_shape, output_shape=output_shape,
            feature_size=feature_size,
            num_channels=num_channels,
            # num_inputs=num_inputs,
            # num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_reset=dilation_reset,
            dropout=dropout,
            causal=causal,
            use_norm=use_norm,
            activation=activation,
            # kernel_initializer=kernel_initializer,
            use_skip_connections=use_skip_connections,
            # input_type=input_type
        )

        input_seqlen, input_size = input_shape
        output_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if num_channels is None:
            num_channels = [feature_size]

        self.input_adapter = None
        self.output_adapter = None

        if input_size != feature_size:
            self.input_adapter = nn.Linear(in_features=input_size, out_features=feature_size)
        if feature_size != output_size:
            self.output_adapter = nn.Linear(in_features=feature_size, out_features=output_size)

        self.tcn = TCN(
            num_inputs=feature_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_reset=dilation_reset,
            dropout=dropout,
            causal=causal,
            use_norm=use_norm,
            activation=activation,
            use_skip_connections=use_skip_connections
        )

    def forward(self, x: Tensor) -> Tensor:
        t = apply_if(x, self.input_adapter)
        t = self.tcn(t)
        y = apply_if(t, self.output_adapter)
        return y
