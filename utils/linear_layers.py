import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from bnn.utils.general import BayesianLayer, kldivergence
from bnn.utils.posteriors import DiagonalNormal
from bnn.utils.priors import GaussianMixture


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BLinear(BayesianLayer):
    """Applies a bayesian linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, weight_prior=None, weight_posterior=None,
                 bias=True, bias_prior=None, bias_posterior=None):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_prior = weight_prior
        self.weight_posterior = weight_posterior
        self.bias_prior = bias_prior
        self.bias_posterior = bias_posterior
        self._init_default_distributions()

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        output = F.linear(x, weights, bias)

        kl = kldivergence(self.weight_prior, self.weight_posterior, weights)
        if self.bias:
            kl = kl + kldivergence(self.bias_prior, self.bias_posterior, bias)

        return output, kl

    def _init_default_distributions(self):
        # specify default priors and variational posteriors
        bound = math.sqrt(2.0 / float(self.in_features))
        dists = {'weight_prior': GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75),
                 'weight_posterior': DiagonalNormal(mean=torch.Tensor(self.out_features,
                                                                      self.in_features).uniform_(-bound, bound),
                                                    rho=torch.Tensor(self.out_features,
                                                                     self.in_features).normal_(-9, 0.001))}
        if self.bias:
            dists['bias_prior'] = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            dists['bias_posterior'] = DiagonalNormal(mean=torch.Tensor(self.out_features).uniform_(-0.01, 0.01),
                                                     rho=torch.Tensor(self.out_features).normal_(-9, 0.001))

        # specify all distributions that are not given by the user as the default distribution
        for d in dists:
            if getattr(self, d) is None:
                setattr(self, d, dists[d])

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
