import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

from stdlib import mul_
from .ts import TimeSeriesModel


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def squeeze_last_dim(tensor: Tensor) -> Tensor:
    # Already converted into 2D tensors
    # if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
    #     return tensor[..., 0]
    return tensor


def seasonality_model(thetas: Tensor, t: ndarray) -> Tensor:
    p = thetas.shape[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(thetas.device))


def trend_model(thetas: Tensor, t: ndarray) -> Tensor:
    p = thetas.shape[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(thetas.device))


def linear_space(backcast_length: int, forecast_length: int, is_forecast=True) -> ndarray:
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.nb_harmonics = nb_harmonics
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_linspace = linear_space(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(backcast_length, forecast_length, is_forecast=True)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(x.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


# ---------------------------------------------------------------------------
# NBeatsNet
# ---------------------------------------------------------------------------
SEASONALITY_BLOCK = 'seasonality'
TREND_BLOCK = 'trend'
GENERIC_BLOCK = 'generic'


class NBeatsNet(nn.Module):

    def __init__(self,
                 backcast_length,
                 forecast_length,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=32,
                 nb_harmonics=None
                 ):

        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        # print('| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)

        # [DEBUG]
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
    # end

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        # print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.backcast_length, self.forecast_length,
                    self.nb_harmonics
                )
                self.parameters.extend(block.parameters())
            # print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        device = backcast.device
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.shape[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(device) - b
                forecast = forecast.to(device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({'value': f.detach().numpy(), 'layer': layer_name})
        return backcast, forecast
    # end
# end


# ---------------------------------------------------------------------------
# TSNBeats
# ---------------------------------------------------------------------------

class TSNBeats(TimeSeriesModel):

    def __init__(self, input_shape, output_shape,
                 hidden_size=32,
                 stack_types=(GENERIC_BLOCK,),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 nb_harmonics=None
                 ):
        super().__init__(input_shape, output_shape,
                         hidden_size=hidden_size,
                         stack_types=stack_types,
                         nb_blocks_per_stack=nb_blocks_per_stack,
                         thetas_dim=thetas_dim,
                         share_weights_in_stack=share_weights_in_stack,
                         nb_harmonics=nb_harmonics
                         )

        self.output_reshape = (-1,) + tuple(output_shape)
        self.nbeats = NBeatsNet(
            backcast_length=mul_(input_shape),
            forecast_length=mul_(output_shape),
            hidden_layer_units=hidden_size,
            stack_types=stack_types,
            nb_blocks_per_stack=nb_blocks_per_stack,
            thetas_dim=thetas_dim,
            share_weights_in_stack=share_weights_in_stack,
            nb_harmonics=nb_harmonics
        )

    def forward(self, x: Tensor) -> Tensor:
        t = torch.flatten(x, start_dim=1)
        backcast, forecast = self.nbeats(t)
        y = torch.reshape(forecast, self.output_reshape)
        return y
# end
