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

class BLinear(BayesianLayer):
    """Applies a bayesian linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, weight_prior=None, weight_posterior=None,
                 bias=True, bias_prior=None, bias_posterior=None):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
                
        if weight_prior is None:
#             self.weight_prior = PriorNormal(loc=0,
#                                      scale=0.5)
            self.weight_prior = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
        else:
            self.weight_prior = weight_prior
        
        if weight_posterior is None:
#             std = 1.0 / math.sqrt(self.out_features)
            std = math.sqrt(6.0 / self.out_features)
            self.weight_posterior = DiagonalNormal(loc=torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1),
                                           rho=torch.Tensor(out_features, in_features).uniform_(-3, -2))
        elif not isinstance(weight_posterior, VariationalDistribution):
            self.log.error('weight_posterior has to be a variational distribution')
            raise ValueError('weight_posterior has to be a variational distribution')
        else:
            self.weight_posterior = weight_posterior
            
        self.bias = bias
        if self.bias:
            if bias_prior is None:
#                 self.bias_prior = PriorNormal(loc=0,
#                                               scale=0.5)
                self.bias_prior = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            else:
                self.bias_prior = bias_prior
        
            if bias_posterior is None:
                std = math.sqrt(6.0 / self.out_features)
                self.bias_posterior = DiagonalNormal(loc=torch.Tensor(out_features).uniform_(-0.1, 0.1),
                                                    rho=torch.Tensor(out_features).uniform_(-3, -2))
            elif not isinstance(bias_posterior, VariationalDistribution):
                self.log.error('bias_posterior has to be a variational distribution')
                raise ValueError('bias_posterior has to be a variational distribution')
            else:
                self.bias_posterior = bias_posterior
    
    def forward(self, input):
        weights = self.weight_posterior.sample()
        
        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None
            
        output = F.linear(input, weights, bias)
                    
        kl = self.weight_posterior.log_prob(weights).sum() - self.weight_prior.log_prob(weights).sum() \
            + self.bias_posterior.log_prob(bias).sum() - self.bias_prior.log_prob(bias).sum()
            
        return output, kl
    
    def extra_repr(self):
        return ('in_features={}, out_features={}, bias={}').format(
            self.in_features, self.out_features, self.bias is not None
        )
        