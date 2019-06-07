import torch
import torch.distributions as dist

class PriorDistribution(dist.Distribution):
    def __init__(self):
        super().__init__()
        
class Normal(PriorDistribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = dist.Normal(loc, scale)
        raise NotImplementedError()
        
    def log_prob(self, value):
        return self.normal.log_prob(value)
    
    def sample(self):