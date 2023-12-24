#
# Autoencoders
#
import torch
import torch.nn as nn
import numpy as np
from ... import nn as nnx


# ---------------------------------------------------------------------------
# AE
# ---------------------------------------------------------------------------

class AE(nn.Module):

    def __init__(self, encoder_net, decoder_net):
        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

    def forward(self, x):
        t = self.encoder(x)
        y = self.decoder(t)
        return y


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(AE):

    def __init__(self, input_size, latent_size):
        super().__init__(
            nnx.Linear(input_size, latent_size),
            nnx.Linear(latent_size, input_size)
        )
        self.input_size = input_size
        self.latent_size = latent_size

    #
    #
    #
    def encode(self, x):
        assert isinstance(x, np.ndarray)
        x = torch.tensor(x)
        with torch.no_grad():
            t = self.encoder(x)
            return t.numpy()

    def decode(self, t):
        assert isinstance(t, np.ndarray)
        t = torch.tensor(t)
        with torch.no_grad():
            x = self.decoder(t)
            return x.numpy()
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
