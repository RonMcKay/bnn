import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import logging

class VariationalDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')
    
    def pdf(self, sample):
        raise NotImplementedError('Probability density function has to be implemented!')
    
    def log_prob(self, sample):
        raise NotImplementedError('Probability density function has to be implemented!')
    
class DiagonalNormal(VariationalDistribution):
    def __init__(self, loc=torch.tensor(0.0), rho=torch.tensor(0.0)):
        super().__init__()
        self.loc = nn.Parameter(loc)
        self.rho = nn.Parameter(rho)
        
    def sample(self):
        std_dev = F.softplus(self.rho).add(1e-6)
        return self.loc + std_dev * torch.randn_like(self.rho)
    
    def pdf(self, sample):
        if sample.size() != self.loc.size():
            raise ValueError('sample does not match with the distribution shape')
        std_dev = F.softplus(self.rho).add(1e-6)
        return sample.sub(self.loc).div(std_dev).pow(2).div(-2.0).exp().div((2*math.pi).sqrt()*std_dev)
    
    def log_prob(self, sample):
        if sample.size() != self.loc.size():
            print(sample.size(), self.loc.size())
            raise ValueError('sample does not match with the distribution shape')
        std_dev = F.softplus(self.rho).add(1e-6)
        log_prob = torch.log(torch.tensor(2.0*math.pi)).div(-2.0) - std_dev.log() - sample.sub(self.loc).div(std_dev).pow(2).div(2.0)
        return log_prob
    
class Uniform(VariationalDistribution):
    def __init__(self, lower_bound=torch.tensor(0.0), upper_bound=torch.tensor(1.0)):
        super().__init__(size)
        self.log = logging.getLogger(__name__)
        self.lower_bound = nn.Parameter(torch.full(self.size, lower_bound))
        self.upper_bound = nn.Parameter(torch.full(self.size, upper_bound))
        
    def sample(self):
        return torch.rand_like(self.lower_bound).mul(self.upper_bound - self.lower_bound).add(self.lower_bound)
    
    def pdf(self, sample):
        if sample.size() != self.lower_bound.size():
            raise ValueError('sample does not match with the distribution shape')
        
        under_lower_bound = sample.lt(self.lower_bound)
        over_upper_bound = sample.gt(self.upper_bound)
        pdf = self.upper_bound.sub(self.lower_bound).reciprocal()
        pdf[under_lower_bound] = torch.tensor(0.0)
        pdf[over_upper_bound] = torch.tensor(0.0)
        return pdf
    
    def log_prob(self, sample):
        if sample.size() != self.lower_bound.size():
            raise ValueError('sample does not match with the distribution shape')
        
        under_lower_bound = sample.lt(self.lower_bound)
        over_upper_bound = sample.gt(self.upper_bound)
        mask = (under_lower_bound + over_upper_bound)>0
        pdf = self.pdf(sample)
        log_prob = torch.zeros_like(pdf)
        log_prob[1-mask] = pdf[1-mask].log()
        if mask.sum() > 0:
            self.log.error('Encountered values outside of distribution. ' \
                           + 'Can not compute log_prob.')
            raise ValueError('Encountered values outside of distribution. ' \
                           + 'Can not compute log_prob.')
        return log_prob
    
class Bernoulli(VariationalDistribution):
    def __init__(self, probs=torch.tensor(0.5)):
        super().__init__(size)
        self.probs = nn.Parameter(probs)
    
class Beta(VariationalDistribution):
    def __init__(self, concentration1 = torch.tensor(0.5),
                 concentration2 = torch.tensor(0.5)):
        super().__init__(size)
        self.concentration1 = nn.Parameter(concentration1)
        self.concentration2 = nn.Parameter(concentration2)