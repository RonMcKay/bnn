import math
import torch
import torch.nn.functional as F
import torch.nn as nn
# import torch.distributions as dist
from torch.distributions import Distribution
from .general import BayesianLayer
from .posteriors import VariationalDistribution
from .posteriors import DiagonalNormal
from .priors import DiagonalNormal as PriorNormal
from .priors import Laplace, GaussianMixture
import logging

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
        self._init_default_distributions()
    
    def forward(self, input, **kwargs):
        weights = self.weight_posterior.sample()
        
        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None
            
        output = F.linear(input, weights, bias)
                    
        kl = self.weight_posterior.log_prob(weights).sum() - self.weight_prior.log_prob(weights).sum()
        if self.bias:
            kl += self.bias_posterior.log_prob(bias).sum() - self.bias_prior.log_prob(bias).sum()
            
        return output, kl
    
    def _init_default_distributions(self):
        # specify default priors and variational posteriors
        dists = {'weight_prior': GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75),
                 'weight_posterior': DiagonalNormal(loc=torch.Tensor(self.out_features,
                                                                     self.in_features).uniform_(-0.1, 0.1),
                                                    rho=torch.Tensor(self.out_features,
                                                                     self.in_features).uniform_(-3, -2))}
        if self.bias:
            dists['bias_prior'] = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            dists['bias_posterior'] = DiagonalNormal(loc=torch.Tensor(self.out_features).uniform_(-0.1, 0.1),
                                                     rho=torch.Tensor(self.out_features).uniform_(-3, -2))
        
        # specify all distributions that are not given by the user as the default distribution
        for d in dists:
            if getattr(self, d) is None:
                setattr(self, d, dists[d])
    
    def extra_repr(self):
        return ('in_features={}, out_features={}, bias={}').format(
            self.in_features, self.out_features, self.bias is not None
        )
        