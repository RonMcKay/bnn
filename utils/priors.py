import torch
import math
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Independent, Normal

class PriorDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        
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
#         normalization = torch.tensor(2.0*math.pi).log() * (-0.5) - self.scale.log()
#         exponential = ((value - self.loc) / self.scale)**2 * (-0.5)
#         return normalization + exponential
        return torch.tensor(2.0*math.pi).log() * (-0.5) - self.scale.log() + ((value - self.loc) / self.scale)**2 * (-0.5)
    
    def extra_repr(self):
        return 'loc={}, scale={}'.format(round(self.loc.item(), 5), round(self.scale.item(), 5))
        
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
    def __init__(self, sigma1, sigma2, pi, mu1=0, mu2=0):
        super().__init__()
        if not isinstance(pi, torch.FloatTensor):
            pi = torch.tensor(pi, dtype=torch.float)
        self.register_buffer('pi', pi.clone())
        
        self.normal1 = DiagonalNormal(loc=mu1, scale=sigma1)
        self.normal2 = DiagonalNormal(loc=mu2, scale=sigma2)
        
    def log_prob(self, value):
        logprob1 = self.normal1.log_prob(value)
        logprob2 = self.normal2.log_prob(value)

        # Numerical stability trick -> unnormalising logprobs will underflow otherwise
        # from: https://github.com/JavierAntoran/Bayesian-Neural-Networks
        max_logprob = torch.max(logprob1, logprob2)
#         normalised_probs = self.pi * torch.exp(logprob1 - max_logprob) + (1-self.pi) * torch.exp(logprob2 - max_logprob)
#         logprob = torch.log(normalised_probs) + max_logprob
#         return logprob
        return torch.log(self.pi * torch.exp(logprob1 - max_logprob) + (1-self.pi) * torch.exp(logprob2 - max_logprob)) + max_logprob