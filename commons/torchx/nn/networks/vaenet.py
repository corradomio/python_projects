import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#
# Note:
#   Autoencoder
#   VariationalAutoencoder
#   MixtureVariationalAutoencoder
#   DirichletVariationalAutoencoder
#

def sq(x): return x*x


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------
# Added:
#   Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
#   https://aclanthology.org/N19-1021.pdf
#
#   https://medium.com/mlearning-ai/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023
#
class BinaryVAELoss(nn.Module):

    def __init__(self, beta=1.):
        """
        Loss used during the training of a LinearVAE with binary data
        (b/w images)

        :param beta: weight to assign to kld
        """
        super().__init__()
        self.beta = beta

    def forward(self, y, x):
        beta = self.beta
        x_hat, mean, log_var = y

        # reproduction loss
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        # Kullback–Leibler divergence
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + beta * kld
# end


class GaussianVAELoss(nn.Module):
    def __init__(self, var=None, sigma=None, beta=1.):
        """
        Loss used during the training of a LinearVAE with gaussian distributed data
        (tabular data)

        :param sigma: standard deviation. Can be a float or an array
        :param beta: weight to assign to kld
        """
        super().__init__()
        self.beta = beta
        var = var if var is not None else 1. if sigma is None else sq(sigma)
        if isinstance(var, np.ndarray):
            self.var = torch.tensor(var)
        else:
            self.var = var

    def forward(self, y, x):
        beta = self.beta
        x_hat, mean, log_var = y

        #
        # variance: 3 sources
        #
        # 1) as parameter   -> no grad
        # 2) computed on x  -> no grad
        # 3) computed on x^ -> grad!

        # reproduction loss
        var = torch.var(x, dim=0)     # not grad
        # var = self.var
        reproduction_loss = ((x_hat - x).pow(2)/var).sum()

        # var = self.var
        # reproduction_loss = 0.5*F.mse_loss(x_hat, x, reduction='sum')/var

        # Kullback–Leibler divergence
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + beta * kld
# end


class LineaVAELoss(nn.Module):

    def __init__(self, beta=1, cycle=None):
        """
        Loss used during the training of LinearVAE

        :param beta: weight to assign to kld
        :param cycle: None or (ramp, const)
                    ramp: how many epochs used in the ramp period
                    const: how many epochs used in the constant period
        """
        super().__init__()
        self.beta = beta
        if cycle is None:
            ramp, const = 0, 0
        else:
            ramp, const = cycle
        self.ramp = ramp
        self.const = const
        self.cycle_id = 0

    def _compute_beta(self):
        if self.ramp == 0:
            return self.beta

        plen = self.cycle_id % (self.ramp + self.const)
        if plen >= self.ramp:
            beta = self.beta
        else:
            beta = self.beta*(plen)/self.ramp

        self.cycle_id += 1
        return beta

    def forward(self, y, x):
        beta = self._compute_beta()
        x_hat, mean, log_var = y

        # SE la media viene fatta da chi usa la loss, allora NON E' QUI che deve essere
        # calcolata la loss media!
        # In questo caso 'reduction' deve essere 'sum' NON 'mean'

        # 'binary cross entropy' VA BENE SOLO PER {0, 1}
        # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x)
        # 'me loss' va bene per valori arbitrari
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

        # questa sembra piu' corretta:
        #   si calcola KL per ogni istanza nel batch (dim=1)
        #   si calcola la media del batch (dim=0)
        # NO: in teoria la media viene fatta a livello piu' alto dividendo per il numero
        #     di elementi nel batch
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + beta*kld
# end


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------

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
        t = self.encoder(x)

        mean = self.mean(t)
        log_var = self.log_var(t)
        z = self.reparameterization(mean, log_var)

        x_hat = self.decoder(z)
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
            t = self.encoder(x)
            mean = self.mean(t)
            log_var = self.log_var(t)
            sigma = torch.exp(0.5 * log_var)
            z = mean + sigma
            # z = self.reparameterization(mean, log_var)
            return z.numpy()

    def decode(self, latent_coords):
        """
        Decode the latent coordinates
        """
        # assert isinstance(latent_coords, np.ndarray)
        if isinstance(latent_coords, np.ndarray):
            latent_coords = torch.tensor(latent_coords)
        with torch.no_grad():
            pred = self.decoder(latent_coords)
            return pred.numpy()
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
