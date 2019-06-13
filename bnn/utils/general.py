import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
#         self.likelihood_cost = nn.CrossEntropyLoss(reduction='none')
        self.likelihood_cost = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, outputs, target, kl, batch_weight):
        loss = []
        for i in range(outputs.size(0)):
            loss.append(self.likelihood_cost(outputs[i], target))
        loss = torch.stack(loss).mean()
        
        return batch_weight * kl + loss