import torch
import math
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Independent, Normal

class PriorDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        
#     def _apply(self, fn):
#         for module in self.children():
#             module._apply(fn)
#         
#         for param in self._parameters.values():
#             if param is not None:
#                 param.data = fn(param.data)
#                 if param._grad is not None:
#                     param._grad.data = fn(param._grad.data)
#                     
#         for key, buf in self._buffers.items():
#             if buf is not None:
#                 self._buffers[key] = fn(buf)
#         self._init_dist()
#         return self
        
class DiagonalNormal(PriorDistribution):
    def __init__(self, loc, scale):
        super().__init__()
        if not isinstance(loc, torch.FloatTensor):
            loc = torch.tensor(loc, dtype=torch.float)
        if not isinstance(scale, torch.FloatTensor):
            scale = torch.tensor(scale, dtype=torch.float)
        self.register_buffer('loc', loc.clone())
        self.register_buffer('scale', scale.clone())
        
    def log_prob(self, value):
        normalization = torch.tensor(2.0*math.pi).log() * (-0.5) - self.scale.log()
        exponential = ((value - self.loc) / self.scale)**2 * (-0.5)
        return normalization + exponential
        
class Laplace(PriorDistribution):
    def __init__(self, loc, scale):
        super().__init__()
        if not isinstance(loc, torch.FloatTensor):
            loc = torch.tensor(loc, dtype=torch.float)
        if not isinstance(scale, torch.FloatTensor):
            scale = torch.tensor(scale, dtype=torch.float)
        self.register_buffer('loc', loc.clone())
        self.register_buffer('scale', scale.clone())
        
    def log_prob(self, value):
        return value.sub(self.loc).abs().div(self.scale).neg() - self.scale.mul(2).log()
        
class GaussianMixture(PriorDistribution):
    """Scale mixture of two Gaussian densities
        from 'Weight Uncertainty in Neural Networks'
    """
    def __init__(self, sigma1, sigma2, pi):
        super().__init__()
        if not isinstance(sigma1, torch.FloatTensor):
            sigma1 = torch.tensor(sigma1, dtype=torch.float)
        if not isinstance(sigma2, torch.FloatTensor):
            sigma2 = torch.tensor(sigma2, dtype=torch.float)
        if not isinstance(pi, torch.FloatTensor):
            pi = torch.tensor(pi, dtype=torch.float)
        self.register_buffer('sigma1', sigma1.clone())
        self.register_buffer('sigma2', sigma2.clone())
        self.register_buffer('pi', pi.clone())
        
        self.normal1 = DiagonalNormal(loc=0, scale=self.sigma1)
        self.normal2 = DiagonalNormal(loc=0, scale=self.sigma2)
        
    def log_prob(self, value):
        logprob1 = self.normal1.log_prob(value)
        logprob2 = self.normal2.log_prob(value)

        # Numerical stability trick -> unnormalising logprobs will underflow otherwise
        # from: https://github.com/JavierAntoran/Bayesian-Neural-Networks
        max_logprob = torch.min(logprob1, logprob2)
        normalised_probs = self.pi + torch.exp(logprob1 - max_logprob) + (1-self.pi) + torch.exp(logprob2 - max_logprob)
        logprob = torch.log(normalised_probs) + max_logprob