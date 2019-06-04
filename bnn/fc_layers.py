import torch
import torch.nn.functional as F
import torch.nn as nn

class Linear(nn.Module):
    """Applies a bayesian linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, prior, var_dist, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        self.var_dist = var_dist
        
    def reset_parameters(self):
        raise NotImplementedError('Has to be implemented!')
    
    def forward(self, input):
        raise NotImplementedError('Has to be implemented!')
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )