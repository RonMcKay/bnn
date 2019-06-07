import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Distribution
from .general import BayesianLayer
from .vardist import VariationalDistribution
import logging

class BLinear(BayesianLayer):
    """Applies a bayesian linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, prior, var_dist, bias=False,
                 bias_prior=None, bias_var_dist=None):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
                
        if not isinstance(prior, Distribution):
            self.log.error('prior has to be a pytorch distribution')
            raise ValueError('prior has to be a pytorch distribution')
        self.prior = prior
        
        if not isinstance(var_dist, VariationalDistribution):
            self.log.error('var_dist has to be a variational distribution')
            raise ValueError('var_dist has to be a variational distribution')
        self.var_dist = var_dist
        self.bias = bias
        
        if not isinstance(bias_prior, Distribution):
            self.log.error('bias_prior has to be a pytorch distribution')
            raise ValueError('bias_prior has to be a pytorch distribution')
        self.bias_prior = bias_prior
        
        if not isinstance(bias_var_dist, VariationalDistribution):
            self.log.error('bias_var_dist has to be a variational distribution')
            raise ValueError('bias_var_dist has to be a variational distribution')
        self.bias_var_dist = bias_var_dist
        
        if self.bias and (self.bias_prior is None or self.bias_var_dist is None):
            self.log.error('If bias is true, bias_prior and bias_var_dist have to be specified.')
        
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