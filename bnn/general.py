import torch
import torch.nn as nn

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.likelihood_cost = nn.CrossEntropyLoss()
        
    def forward(self, output, target, kl, batch_weight):
        loss = self.likelihood_cost(output, target)
        
        return batch_weight * kl + loss