import torch.nn as nn
import torch.nn.functional as F
from ... import nn as nnx
from .lin import Dense


class MixtureDensityNetwork(nnx.MixtureDensityNetwork):

    def __init__(self, input, units, **kwargs):
        super().__init__(in_features=input,
                         out_features=units,
                         **kwargs)


# ---------------------------------------------------------------------------
# MDN
# ---------------------------------------------------------------------------

EPSILON = 1e-10


def elu_plus_one_plus_epsilon(x):
    return F.elu(x) + 1 + EPSILON


class MDN(nn.Module):

    def __init__(self, input, units, n_mixtures):
        super().__init__()
        self.input = input
        self.units = units
        self.n_mixtures = n_mixtures

        self.mus = Dense(input=input, units=units*n_mixtures)
        self.sigma = Dense(input=units*n_mixtures, units=units*n_mixtures)
        self.activation = elu_plus_one_plus_epsilon
        self.pi = Dense(input=units*n_mixtures, units=n_mixtures)

    def forward(self, inputs):
        t = inputs
        t = self.mus(t)
        t = self.sigma(t)
        t = self.activation(t)
        t = self.pi(t)
        return t
