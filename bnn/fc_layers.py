import math
import torch
import torch.nn.functional as F
import torch.nn as nn
# import torch.distributions as dist
from torch.distributions import Distribution
from .general import BayesianLayer
from .vardist import VariationalDistribution
from .vardist import DiagonalNormal
from .priordist import DiagonalNormal as PriorNormal
import logging

class BLinear(BayesianLayer):
    """Applies a bayesian linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, prior=None, var_dist=None,
                 bias=True, bias_prior=None, bias_var_dist=None):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
                
        if prior is None:
            self.prior = PriorNormal(loc=torch.zeros((out_features, in_features)),
                                     scale=torch.ones((out_features, in_features)))
        elif not isinstance(prior, Distribution):
            self.log.error('prior has to be a pytorch distribution')
            raise ValueError('prior has to be a pytorch distribution')
        else:
            self.prior = prior
        
        if var_dist is None:
            std = 1.0 / math.sqrt(self.out_features)
            self.var_dist = DiagonalNormal(mean=torch.Tensor(out_features, in_features).uniform_(-std, std),
                                           rho=torch.full((out_features, in_features), 0.05))
        elif not isinstance(var_dist, VariationalDistribution):
            self.log.error('var_dist has to be a variational distribution')
            raise ValueError('var_dist has to be a variational distribution')
        else:
            self.var_dist = var_dist
            
        self.bias = bias
        if self.bias:
            if bias_prior is None:
                self.bias_prior = PriorNormal(loc=torch.zeros((out_features,)),
                                              scale=torch.full((out_features,), 1.05))
            elif not isinstance(bias_prior, Distribution):
                self.log.error('bias_prior has to be a pytorch distribution')
                raise ValueError('bias_prior has to be a pytorch distribution')
            else:
                self.bias_prior = bias_prior
        
            if bias_var_dist is None:
                std = 1.0 / math.sqrt(self.out_features)
                self.bias_var_dist = DiagonalNormal(mean=torch.Tensor(out_features).uniform_(-std, std),
                                                    rho=torch.full((out_features,), 0.05))
            elif not isinstance(bias_var_dist, VariationalDistribution):
                self.log.error('bias_var_dist has to be a variational distribution')
                raise ValueError('bias_var_dist has to be a variational distribution')
            else:
                self.bias_var_dist = bias_var_dist
        
    def reset_parameters(self):
        raise NotImplementedError('Has to be implemented!')
    
    def forward(self, input):
        # sample weights and bias
        weights = self.var_dist.sample()
        
        if self.bias:
            bias = self.bias_var_dist.sample()
            
        output = F.linear(input, weights, bias)
        
        kl = (self.var_dist.log_prob(weights) - self.prior.log_prob(weights)).sum() \
            + (self.bias_var_dist.log_prob(bias) - self.bias_prior.log_prob(bias)).sum()
            
        return output, kl
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )