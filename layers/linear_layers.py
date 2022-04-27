# Standard Library
import logging

# Thirdparty libraries
import torch
import torch.nn.functional as F

# Firstparty libraries
from bnn import BayesianLayer
from bnn.distributions.posteriors import DiagonalNormal
from bnn.distributions.priors import GaussianMixture
from bnn.utils import kldivergence


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
        dists = {'weight_prior': GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75),
                 'weight_posterior': DiagonalNormal(mean=torch.Tensor(self.out_features,
                                                                      self.in_features).normal_(0, 0.1),
                                                    rho=torch.Tensor(self.out_features,
                                                                     self.in_features).normal_(-5, 0.1))}
        if self.bias:
            dists['bias_prior'] = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            dists['bias_posterior'] = DiagonalNormal(mean=torch.Tensor(self.out_features).normal_(0, 0.1),
                                                     rho=torch.Tensor(self.out_features).normal_(-5, 0.1))

        # specify all distributions that are not given by the user as the default distribution
        for d in dists:
            if getattr(self, d) is None:
                setattr(self, d, dists[d])

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
