import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from bnn.utils.general import BayesianLayer
from bnn.utils.posteriors import DiagonalNormal
from bnn.utils.priors import DiagonalNormal as PriorNormal
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
        self.log = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_prior = weight_prior
        self.weight_posterior = weight_posterior
        self.bias_prior = bias_prior
        self.bias_posterior = bias_posterior
        self.kl_weights_closed = False
        self.kl_bias_closed = False
        self._init_default_distributions()

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        output = F.linear(x, weights, bias)

        # Calculate KL in closed form if prior and posterior are normal distribution
        if self.kl_weights_closed:
            kl = self.closed_form_kl()
        else:
            kl = self.weight_posterior.log_prob(weights).sum() - self.weight_prior.log_prob(weights).sum()
        if self.bias:
            if self.kl_bias_closed:
                kl = kl + self.closed_form_kl(bias=True)
            else:
                kl = kl + self.bias_posterior.log_prob(bias).sum() - self.bias_prior.log_prob(bias).sum()

        return output, kl

    def closed_form_kl(self, bias=False):
        if bias:
            sigma_prior = self.bias_prior.get_std()
            sigma_posterior = self.bias_posterior.get_std()
            mean_prior = self.bias_prior.get_mean()
            mean_posterior = self.bias_posterior.get_mean()
        else:
            sigma_prior = self.weight_prior.get_std()
            sigma_posterior = self.weight_posterior.get_std()
            mean_prior = self.weight_prior.get_mean()
            mean_posterior = self.weight_posterior.get_mean()

        return (torch.log(sigma_prior / sigma_posterior) + (
                sigma_posterior ** 2 + (mean_posterior - mean_prior) ** 2) / (2 * sigma_prior ** 2) - 0.5).sum()

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

        if isinstance(self.weight_prior, PriorNormal) and isinstance(self.weight_posterior, DiagonalNormal):
            self.kl_weights_closed = True
            self.log.debug('Kullback Leibler Divergence for weights will be calculated in closed form.')

        if isinstance(self.bias_prior, PriorNormal) and isinstance(self.bias_posterior, DiagonalNormal):
            self.kl_bias_closed = True
            self.log.debug('Kullback Leibler Divergence for biases will be calculated in closed form.')

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
