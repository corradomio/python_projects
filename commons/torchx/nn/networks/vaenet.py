import numpy as np
import torch
import torch.nn as nn
from timing import tprint

from ... import nn as nnx


#
# Note:
#   Autoencoder
#   VariationalAutoencoder
#   MixtureVariationalAutoencoder
#   DirichletVariationalAutoencoder
#


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------

class LinearVAELoss(nn.Module):

    def __init__(self, beta=1):
        """
        Loss used during the training of LinearVAE

        :param beta: weight to assign to kld
        """
        super().__init__()
        self.beta = beta

    def forward(self, y, x):
        beta = self.beta
        x_hat, mean, log_var = y

        # 'binary cross entropy' VA BENE SOLO PER {0, 1}
        # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x)
        # 'me loss' va bene per valori arbitrari
        reproduction_loss = nn.functional.mse_loss(x_hat, x)

        # kld2 = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # questa sembra piu' corretta:
        #   si calcola KL per ogni istanza nel batch (dim=1)
        #   si calcola la media del batch (dim=0)
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)

        # tprint(f"{reproduction_loss:.4}, {kld:.4}")
        return reproduction_loss + beta*kld
# end


class LinearVAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size=2):
        """

        :param input_size: input size
        :param hidden_size: hidden size
        :param latent_size: gaussian latent size
        :param vae: [debug] if to use VAE or plain AE
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
        )

        self.mean    = nn.Linear(hidden_size, latent_size, bias=False)
        self.log_var = nn.Linear(hidden_size, latent_size, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            # nn.Sigmoid()
            # nn.ReLU(),
        )
    # end

    def forward(self, x):
        # assert 0. <= x.min() <= x.max() <= 1., "The data must be in range [0, 1]"
        t = self.encoder.forward(x)

        mean    = self.mean.forward(t)
        log_var = self.log_var.forward(t)

        z = self.reparameterization(mean, log_var)

        x_hat = self.decoder.forward(z)
        return x_hat, mean, log_var

    @staticmethod
    def reparameterization(mean, log_var):
        # eps
        epsilon = torch.randn_like(log_var)
        # var = sigma^2 -> log(var) = log(sigma^2) -> sigma = sqrt(exp(log(sigma^2)))
        sigma = torch.exp(0.5 * log_var)
        z = mean + sigma * epsilon
        return z

    #
    # To use in the code
    #
    def encode(self, x):
        """
        Encode x in the latent space coordinates
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        with torch.no_grad():
            t = self.encoder.forward(x)
            mean = self.mean.forward(t)
            log_var = self.log_var.forward(t)
            z = self.reparameterization(mean, log_var)
            return z.numpy()

    def decode(self, latent_coords):
        """
        Decode the latent coordinates
        """
        # assert isinstance(latent_coords, np.ndarray)
        if isinstance(latent_coords, np.ndarray):
            latent_coords = torch.tensor(latent_coords)
        with torch.no_grad():
            pred = self.decoder.forward(latent_coords)
            return pred.numpy()

    def loss(self):
        return LinearVAELoss()
# end


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = nnx.Linear(input_size, latent_size)
        self.decoder = nnx.Linear(latent_size, input_size)

    def forward(self, x):
        t = self.encoder.forward(x)
        y = self.decoder.forward(t)
        return y

    #
    #
    #
    def encode(self, x):
        assert isinstance(x, np.ndarray)
        x = torch.tensor(x)
        with torch.no_grad():
            t = self.encoder.forward(x)
            return t.numpy()

    def decode(self, t):
        assert isinstance(t, np.ndarray)
        t = torch.tensor(t)
        with torch.no_grad():
            x = self.decoder.forward(t)
            return x.numpy()
# end
