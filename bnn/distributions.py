import torch
import torch.nn.functional as F
import torch.nn as nn

class VariationalDistribution(nn.Module):
    def __init__(self, size):
        self.size = size
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')
    
class DiagonalNormal(VariationalDistribution):
    def __init__(self, size):
        super().__init__(size)
        self.mean = nn.Parameter(torch.Tensor(self.size))
        self.std_dev = nn.Parameter(torch.Tensor(self.size))
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')
    
class Uniform(VariationalDistribution):
    def __init__(self, size):
        super().__init__(size)
        self.lower_bound = nn.Parameter(torch.Tensor(self.size))
        self.upper_bound = nn.Parameter(torch.Tensor(self.size))
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')
    
class Bernoulli(VariationalDistribution):
    def __init__(self, size):
        super().__init__(size)
        self.probs = nn.Parameter(torch.Tensor(self.size))
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')
    
class Beta(VariationalDistribution):
    def __init__(self, size):
        super().__init__(size)
        self.concentration1 = nn.Parameter(torch.Tensor(self.size))
        self.concentration2 = nn.Parameter(torch.Tensor(self.size))
        
    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')