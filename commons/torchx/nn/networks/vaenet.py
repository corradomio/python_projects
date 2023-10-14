import torch
import torch.nn as nn

from ..modules.lin import Linear


class Sampler(nn.Module):

    def forward(self, inputs):
        # mean, log_var = inputs
        mean = inputs[:, 0]
        log_var = inputs[:, 1]
        epsilon = torch.normal(mean)
        return mean + torch.exp(.5*log_var) * epsilon


class LinearVAE(nn.Module):

    def __init__(self, in_features, n_mixtures):
        super().__init__()
        self.in_features = in_features
        self.n_mixtures = n_mixtures

        # input -> (n_mixtures, [mean, log_var])
        self.encoder = Linear(in_features=in_features, out_features=(2, n_mixtures))
        self.sampler = Sampler()
        self.decoder = Linear(in_features=n_mixtures, out_features=in_features)
    # end

    def forward(self, inputs):
        t = inputs
        t = self.encoder.forward(t)
        t = self.sampler.forward(t)
        t = self.decoder.forward(t)
        return t


