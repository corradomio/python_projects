import torch
import torch.distributions as D


class MultivariateNormalDiag(D.MultivariateNormal):

    def __init__(self, loc, covariance_diag):
        super().__init__(loc, torch.diag(covariance_diag))

