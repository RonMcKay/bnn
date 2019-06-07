import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Independent, Normal

class PriorDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        
        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
                    
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        self._init_dist()
        return self
        
class DiagonalNormal(PriorDistribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.register_buffer('loc', loc.clone())
        self.register_buffer('scale', scale.clone())
        self._init_dist()
        
    def log_prob(self, value):
        return self.normal.log_prob(value)
    
    def sample(self):
        return self.normal.sample()
        
    def _init_dist(self):
        self.normal = Independent(Normal(loc=self.loc,
                                         scale=self.scale),
                                  len(self.loc.shape))
        
class GaussianMixture(PriorDistribution):
    """Scale mixture of two Gaussian densities
        from 'Weight Uncertainty in Neural Networks'
    """
    def __init__(self, sigma1, sigma2, pi):
        super().__init__()
        self.register_buffer('sigma1', sigma1.clone())
        self.register_buffer('sigma2', sigma2.clone())
        self.register_buffer('pi', pi.clone())
        
    def log_prob(self, value):
        return self.pi * self.normal1.log_prob(value) \
            + (1-self.pi) * self.normal2.log_prob(value)
            
    def sample(self):
        return self.pi * self.normal1.sample() \
            + (1-self.pi) * self.normal2.sample()
    
    def _init_dist(self):
        self.normal1 = Independent(Normal(loc=torch.zeros_like(self.sigma1),
                                          scale=self.sigma1),
                                   len(self.sigma1.shape))
        self.normal2 = Independent(Normal(loc=torch.zeros_like(self.sigma2),
                                          scale=self.sigma2),
                                   len(self.sigma2.shape))