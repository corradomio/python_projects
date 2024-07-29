# Other implementations
#
#   https://github.com/sagelywizard/pytorch-mdn
#
from enum import Enum, auto

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from .module import Module
from . import lin as nnx
from stdlib import mul_


# ---------------------------------------------------------------------------
# from keras-mdn-layer package
# ---------------------------------------------------------------------------
# https://github.com/cpmpercussion/keras-mdn-layer
#
EPS = 1e-6

# Non Negative Exponential Linear Unit (nnelu)
def elu_plus_one_plus_epsilon(x):
    return F.elu(x) + 1 + EPS


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.

    Arguments:
    w -- a list or numpy array of logits

    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = w / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    # e = w/t
    # e -= e.max()
    # e = torch.exp(e)
    # dist = e/e.sum()
    return dist


def sample_from_categorical(dist):
    """Samples from a categorical model PDF.

    Arguments:
    dist -- the parameters of the categorical model

    Returns:
    One sample from the categorical model, or -1 if sampling fails.
    """
    # r = torch.rand(1)  # uniform random number in [0,1]
    # accumulate = torch.tensor(0., dtype=dist.dtype)
    # for i in range(0, len(dist)):
    #     accumulate += dist[i]
    #     if accumulate >= r:
    #         return i
    # return -1
    n = len(dist)
    r = np.random.rand(1)
    accumulate = 0.
    for i in range(0, n):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    return n-1


def split_mixture_params(mdn_params, output_size, n_mixtures):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    if len(mdn_params.shape) == 1:
        mus = mdn_params[:n_mixtures*output_size]
        sigmas = mdn_params[n_mixtures*output_size:2*n_mixtures*output_size]
        pi_logits = mdn_params[-n_mixtures:]
    else:
        mus = mdn_params[:, :n_mixtures * output_size]
        sigmas = mdn_params[:, n_mixtures*output_size:2*n_mixtures*output_size]
        pi_logits = mdn_params[:, -n_mixtures:]

        mus = mus.reshape((-1, n_mixtures, output_size))
        sigmas = sigmas.reshape((-1, n_mixtures, output_size))

    return mus, sigmas, pi_logits


def sample_from_output(params, output_size, n_mixtures, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented

    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)

    Returns:
    One sample from the the mixture model.
    """
    mus, sigmas, pi_logits = split_mixture_params(params, output_size, n_mixtures)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_size:(m + 1) * output_size]
    sig_vector = sigmas[m * output_size:(m + 1) * output_size]
    scale_matrix = np.diag(sig_vector)  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(np.array(mus_vector), np.array(cov_matrix), 1)
    return sample


# def get_keras_mixture_loss_func(output_size, n_mixtures):
#     def mdn_loss_func(y_true, mdn_params):
#         # Reshape inputs in case this is used in a TimeDistributed layer
#         # UPDATE: instead to return a conactenated tensor, NOW it is returned just the tuple
#         y_true = y_true.reshape((-1, output_size))
#         mus, sigmas, pi = mdn_params
#
#         cat = Categorical(logits=pi)
#
#         component_splits = [output_size] * n_mixtures
#         mus = torch.tensor_split(mus, component_splits, dim=1)
#         sigmas = torch.tensor_split(sigmas, component_splits, dim=1)
#
#         coll = [MultivariateNormal(loc=loc, scale_tril=scale) for loc, scale in zip(mus, sigmas)]
#         mixture = MixtureSameFamily(cat, coll)
#
#         loss = mixture.log_prob(y_true)
#         loss = torch.negative(loss)
#         loss = torch.mean(loss)
#
#         return loss
#
#     return mdn_loss_func


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class MixtureDensityNetwork(Module):

    def __init__(self, in_features, out_features, n_mixtures,):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_mixtures = n_mixtures

        self.mus = nnx.Linear(in_features=in_features, out_features=(n_mixtures * out_features))
        self.sigmas = nnx.Linear(in_features=in_features, out_features=(n_mixtures * out_features))
        self.pi_logits = nn.Linear(in_features=in_features, out_features=n_mixtures)

    def forward(self, x):
        mus = self.mus(x)
        sigmas = elu_plus_one_plus_epsilon(self.sigmas(x))
        pi_logits = self.pi_logits(x)

        #   mus     [n_mixtures, output_size]
        #   sigmas  [n_mixtures, output_size]
        #   pi      [n_mixtures]

        return torch.cat([mus, sigmas, pi_logits], dim=1)
# end


class MixtureDensityNetworkLoss(Module):

    def __init__(self, out_features, n_mixtures):
        super().__init__()
        self.out_features = out_features
        self.n_mixtures = n_mixtures

    def forward(self, mdn_params, y_true):
        out_features = self.out_features
        n_mixtures = self.n_mixtures

        #   mus     [N, n_mixtures, output_size]
        #   sigmas  [N, n_mixtures, output_size]
        #   pi      [N, n_mixtures]

        # Reshape inputs in case this is used in a TimeDistributed layer
        # UPDATE: instead to return a concatenated tensor, NOW it is returned just the tuple
        batch = y_true.shape[0]
        y_true = y_true.reshape((batch, -1))
        mus, sigmas, pi_logits = split_mixture_params(mdn_params, out_features, n_mixtures)
        cat = D.Categorical(logits=pi_logits)
        total_loss = None

        for i in range(out_features):
            mu = mus[:, :, i]
            sigma = sigmas[:, :, i]
            norm = D.Normal(mu, sigma)
            mixture = D.MixtureSameFamily(cat, norm)
            loss = mixture.log_prob(y_true[:, i])
            loss = torch.negative(loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss.mean()
# end


class MixtureDensityNetworkPredictor:

    def __init__(self, model, out_features, n_mixtures, n_samples=1):
        self.model = model
        self.out_features = mul_(out_features)
        self.n_mixtures = n_mixtures
        self.n_samples = n_samples

    def predict(self, x):
        n = len(x)
        pred_list = []
        for i in range(n):
            pred = self._predict(x[i:i+1])
            pred_list.append(pred)
        return np.concatenate(pred_list, axis=0)

    def _predict(self, x):
        output_size = self.out_features
        n_mixtures = self.n_mixtures
        pred_fscaled = self.model.predict(x)
        pred_dist = np.zeros((1, 1, self.out_features))
        for i in range(self.n_samples):
            y_samples = np.apply_along_axis(sample_from_output, 1, pred_fscaled, output_size, n_mixtures, temp=1.0)
            pred_dist += y_samples
        return pred_dist.mean(axis=0)
# end


# ---------------------------------------------------------------------------
#   https://eadains.github.io/OptionallyBayesHugo/posts/vol_mdn/
#
# Implementation
#
#   https://github.com/tonyduan/mixture-density-network
#
#

class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class TonyduanMixtureDensityNetwork(Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    in_feature: int; dimensionality of the covariates
    out_features: int; dimensionality of the response variable
    n_mixtures: int; number of components in the mixture model
    """
    def __init__(self, in_features, out_features, n_mixtures,
                 hidden_size=None,
                 noise_type=NoiseType.DIAGONAL, fixed_noise_level=None, eps=1e-6):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: out_features * n_mixtures,
            NoiseType.ISOTROPIC: n_mixtures,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]

        if hidden_size is None:
            hidden_size = n_mixtures*out_features

        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_mixtures
        self.noise_type = noise_type
        self.fixed_noise_level = fixed_noise_level
        self.eps = eps

        self.pi_network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_mixtures),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features * n_mixtures + num_sigma_channels)
        )

    def forward(self, x):
        eps = self.eps
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.out_features * self.n_components]
        sigma = normal_params[..., self.out_features * self.n_components:]

        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.out_features)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.out_features)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)

        mu = mu.reshape(-1, self.n_components, self.out_features)
        sigma = sigma.reshape(-1, self.n_components, self.out_features)
        return log_pi, mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def loss_hat(self, y_hat, y):
        log_pi, mu, sigma = y_hat
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik.mean()

    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples
# end


class TonyduanMixtureDensityNetworkLoss(Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mdn_params, y_true):
        log_pi, mu, sigma = mdn_params
        z_score = (y_true.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik.mean()
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
